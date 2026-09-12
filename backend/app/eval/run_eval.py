"""
Evaluation harness.

Two layers, run separately since one is free and one costs API calls:

1. `eval_severity()` -- checks the deterministic status/severity rule
   (severity.py) against hand-labeled expected outputs. No API calls,
   runs instantly, safe to run on every change.

2. `eval_retrieval()` -- checks retrieval quality (Recall@K, Precision@K,
   MRR) against a hand-labeled query -> correct-source set, split by
   difficulty (exact / synonym / paraphrase). No API calls. This is the
   baseline we compare against before/after any retrieval upgrade
   (embeddings, hybrid, reranking) -- see retrieval_test_cases.json.

3. `eval_generation()` -- runs the full retrieval + generation pipeline
   on a couple of cases and does light checks (does every finding cite a
   source, does it avoid banned words like specific drug names). Uses the
   real Gemini API, so it costs a few free-tier calls per run.

Run with:  python -m app.eval.run_eval
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # backend/ on path

from app.schemas import LabValue
from app.severity import score

TEST_CASES_PATH = Path(__file__).parent / "test_cases.json"
RETRIEVAL_TEST_CASES_PATH = Path(__file__).parent / "retrieval_test_cases.json"

# Words that should never appear in "next_step" -- if they do, the model
# is suggesting specific treatment, which the prompt explicitly forbids.
BANNED_NEXT_STEP_WORDS = [
    "mg", "dose", "dosage", "metformin", "statin", "insulin",
    "take ", "prescribe", "start taking",
]


def eval_severity():
    cases = json.loads(TEST_CASES_PATH.read_text())
    total, correct_status, correct_severity = 0, 0, 0
    failures = []

    for case in cases:
        for row in case["values"]:
            lv = LabValue(
                test=row["test"],
                value=row["value"],
                unit=row["unit"],
                reference_range=row["reference_range"],
                reported_flag=row["reported_flag"],
            )
            scored = score(lv)
            total += 1

            exp_status = case["expected_status"][row["test"]]
            exp_severity = case["expected_severity"][row["test"]]

            if scored.status == exp_status:
                correct_status += 1
            else:
                failures.append(
                    f"[{case['name']}] {row['test']}: status expected "
                    f"'{exp_status}' got '{scored.status}'"
                )

            if scored.severity == exp_severity:
                correct_severity += 1
            else:
                failures.append(
                    f"[{case['name']}] {row['test']}: severity expected "
                    f"'{exp_severity}' got '{scored.severity}' "
                    f"(deviation={scored.deviation_pct}%)"
                )

    print("=== Severity Rule Eval (free, deterministic) ===")
    print(f"Status accuracy:   {correct_status}/{total} ({100*correct_status/total:.0f}%)")
    print(f"Severity accuracy: {correct_severity}/{total} ({100*correct_severity/total:.0f}%)")
    if failures:
        print("\nFailures:")
        for f in failures:
            print(" -", f)
    print()
    return correct_status == total and correct_severity == total


def eval_retrieval(k: int = 3, retrieve_fn=None, label: str = "BM25 (current)"):
    """
    Retrieval-only eval. Decoupled from generation on purpose: this must
    keep working (and stay comparable) as we swap in embeddings / hybrid /
    reranking in later steps, without needing an API key.

    retrieve_fn: callable(query: str, top_k: int) -> list of objects with
    a `.source` attribute, ranked best-first. Defaults to the current
    BM25 retriever so this is runnable today with zero changes elsewhere.
    """
    if retrieve_fn is None:
        from app.retrieval import get_knowledge_base
        kb = get_knowledge_base()
        retrieve_fn = lambda query, top_k: kb.search(query, top_k=top_k)

    cases = json.loads(RETRIEVAL_TEST_CASES_PATH.read_text())

    by_difficulty = {}  # difficulty -> [hit@k bools]
    reciprocal_ranks = []
    precision_at_k_scores = []
    misses = []

    for case in cases:
        query = case["query"]
        expected = case["expected_source"]
        difficulty = case.get("difficulty", "unlabeled")

        results = retrieve_fn(query, k)
        retrieved_sources = [r.source for r in results]

        hit = expected in retrieved_sources
        by_difficulty.setdefault(difficulty, []).append(hit)

        # Precision@K: of the K slots returned, how many are the correct doc.
        # With exactly one correct doc per query, this is 1/len(results) if
        # hit else 0 -- still worth tracking as len(results) shrinks (e.g.
        # BM25 returning zero-score docs get filtered out already).
        if results:
            precision_at_k_scores.append(
                sum(1 for s in retrieved_sources if s == expected) / len(results)
            )
        else:
            precision_at_k_scores.append(0.0)

        # MRR: 1/rank of first correct hit, 0 if not found in top K.
        rr = 0.0
        for rank, source in enumerate(retrieved_sources, start=1):
            if source == expected:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        if not hit:
            misses.append(
                f"[{difficulty}] '{query}' -> expected '{expected}', "
                f"got {retrieved_sources or '(nothing retrieved)'}"
            )

    total = len(cases)
    overall_recall = sum(sum(v) for v in by_difficulty.values()) / total
    mrr = sum(reciprocal_ranks) / total
    mean_precision = sum(precision_at_k_scores) / total

    print(f"=== Retrieval Eval: {label} (K={k}, free, deterministic) ===")
    print(f"Recall@{k}:    {overall_recall*100:.0f}%  ({sum(sum(v) for v in by_difficulty.values())}/{total})")
    print(f"MRR:          {mrr:.2f}")
    print(f"Precision@{k}: {mean_precision*100:.0f}%")
    print("\nRecall@%d by difficulty:" % k)
    for difficulty, hits in by_difficulty.items():
        print(f"  {difficulty:12s} {sum(hits)}/{len(hits)} ({100*sum(hits)/len(hits):.0f}%)")
    if misses:
        print("\nMisses:")
        for m in misses:
            print(" -", m)
    print()
    return {"recall_at_k": overall_recall, "mrr": mrr, "precision_at_k": mean_precision}


def eval_embedding_retrieval(k: int = 3):
    """
    Same eval set and metrics as eval_retrieval(), but against the Jina
    embeddings retriever instead of BM25. Requires JINA_API_KEY (free
    tier). Run separately (--with-embeddings) since it costs real API
    calls, unlike the BM25 eval.
    """
    from app.embedding_retrieval import get_embedding_knowledge_base

    kb = get_embedding_knowledge_base()
    return eval_retrieval(
        k=k,
        retrieve_fn=lambda query, top_k: kb.search(query, top_k=top_k),
        label="Jina embeddings",
    )


def eval_hybrid_retrieval(k: int = 3):
    """
    Same eval set and metrics, against hybrid (BM25 + Jina embeddings via
    RRF) retrieval. Requires JINA_API_KEY -- gated together with the
    embeddings eval since hybrid depends on it.
    """
    from app.hybrid_retrieval import search as hybrid_search

    return eval_retrieval(k=k, retrieve_fn=hybrid_search, label="Hybrid (BM25 + Jina, RRF)")


def eval_generation():
    from app.generation import generate_report

    cases = json.loads(TEST_CASES_PATH.read_text())
    print("=== Generation Eval (uses real Groq API calls) ===")

    total_findings, findings_with_sources, banned_word_hits = 0, 0, 0

    for case in cases:
        scored_values = [
            score(
                LabValue(
                    test=r["test"], value=r["value"], unit=r["unit"],
                    reference_range=r["reference_range"], reported_flag=r["reported_flag"],
                )
            )
            for r in case["values"]
        ]
        try:
            report = generate_report(scored_values)
        except Exception as e:
            print(f"[{case['name']}] generation failed: {e}")
            continue

        for finding in report.findings:
            total_findings += 1
            if finding.sources:
                findings_with_sources += 1
            text = (finding.next_step or "").lower()
            if any(w in text for w in BANNED_NEXT_STEP_WORDS):
                banned_word_hits += 1
                print(f"  [{case['name']}] BANNED WORD in next_step for {finding.test}: {finding.next_step}")

        expected_tests_norm = {v["test"].strip().lower() for v in case["values"]}
        returned_tests = [f.test for f in report.findings]
        # Same substring-fallback logic as _find_sources() in generation.py --
        # matches abbreviation expansions like "WBC" -> "White Blood Cell
        # Count (WBC)" so this diagnostic doesn't raise false alarms for
        # cases the real citation lookup already handles correctly.
        unexpected = [
            t for t in returned_tests
            if not any(e in t.strip().lower() or t.strip().lower() in e for e in expected_tests_norm)
        ]
        print(f"[{case['name']}] {len(report.findings)} findings generated "
              f"({returned_tests}), bottom_line: {report.bottom_line[:80]}...")
        if unexpected:
            print(f"  ^ WARNING: findings not matching any input test name: {unexpected}")

    print()
    print(f"Findings with at least one cited source: {findings_with_sources}/{total_findings}")
    print(f"Banned next-step words found: {banned_word_hits}")


if __name__ == "__main__":
    severity_ok = eval_severity()
    print(f"Severity eval {'PASSED' if severity_ok else 'HAD FAILURES'}\n")

    eval_retrieval()

    if "--with-embeddings" in sys.argv:
        eval_embedding_retrieval()
        eval_hybrid_retrieval()
    else:
        print("Skipping embedding + hybrid retrieval evals (need JINA_API_KEY). "
              "Run with --with-embeddings to include them.\n")

    if "--with-generation" in sys.argv:
        eval_generation()
    else:
        print("Skipping generation eval (needs GROQ_API_KEY). "
              "Run with --with-generation to include it.")
