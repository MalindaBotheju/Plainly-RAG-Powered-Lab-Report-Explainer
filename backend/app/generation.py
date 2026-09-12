"""
Generation step.

Takes the rule-scored lab values (status/severity already computed
deterministically -- see severity.py) plus retrieved reference passages,
and asks Gemini to write the final plain-English report. The LLM's job
here is narrow and constrained on purpose:

  - It does NOT decide severity (already computed).
  - It does NOT invent next steps beyond "monitor / repeat test / see a
    doctor" (enforced by the prompt).
  - It must ground each explanation in the retrieved passage for that
    test, not general knowledge, to reduce hallucination.
  - It must write in plain English, short bullets, no paragraphs.

Retrieval: uses embedding_retrieval.py (Jina embeddings), not the
original BM25 retriever (retrieval.py) or hybrid RRF (hybrid_retrieval.py)
-- embeddings alone scored best in the eval harness (100% recall vs.
BM25's 90%; hybrid actually underperformed embeddings alone at this small
knowledge-base size). See eval/run_eval.py and its docstrings for the
comparison and why.

LLM: uses Groq (see config.GROQ_TEXT_MODEL), not Gemini -- switched for a
much higher free-tier daily quota and to drop a deprecated SDK dependency
(google-generativeai).
"""
import json
import re
from typing import List, Optional

from groq import Groq, RateLimitError

from . import config
from .embedding_retrieval import retrieve_for_test
from .schemas import Finding, ReportResult, ScoredLabValue


class GenerationRateLimitError(RuntimeError):
    """Raised when Groq's free-tier quota (per-minute or per-day) is hit."""

DISCLAIMER = (
    "This is an AI-generated summary for informational purposes only. "
    "It is not a medical diagnosis. Please review these results with a "
    "licensed doctor."
)

_SYSTEM_INSTRUCTIONS = """You explain lab results to patients in plain English.

Hard rules:
- Use short bullet points, never long paragraphs.
- Assume the reader has no medical background. Never use medical jargon
  without immediately explaining it in plain words.
- Base every "what it means" explanation ONLY on the reference passage
  provided for that test. If the passage doesn't cover something, do not
  make it up.
- If a test's REFERENCE PASSAGE says no reference material is available,
  do not write a vague placeholder like "no information provided." Say
  plainly, in one short sentence, that this specific test isn't in your
  reference library yet and the patient should ask their doctor directly
  about it -- this is a real, honest limitation, not an error to hide.
- "Next step" suggestions are limited to: getting a follow-up/repeat test,
  monitoring, or seeing a doctor. NEVER suggest a specific medication,
  dosage, or specific treatment.
- Do not diagnose. You may describe what a pattern "can suggest" or "is
  associated with," never state a diagnosis as fact.
- Keep each explanation to 1-2 short sentences.

Return ONLY a JSON object with this exact shape, no markdown fences, no
extra commentary:
{
  "overall_summary": "1-2 sentence plain-English summary of how many findings are off and the general theme",
  "findings": [
    {
      "test": "...",
      "value": "...",
      "unit": "...",
      "status": "high" | "low",
      "severity": "mild" | "concerning",
      "plain_meaning": "...",
      "next_step": "..."
    }
  ],
  "bottom_line": "2-3 sentence plain-English takeaway a patient should walk away with"
}
"""


def _get_client() -> Groq:
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys and set it in your .env "
            "(local) or Render environment variables (deployed)."
        )
    return Groq(api_key=config.GROQ_API_KEY)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _normalize_test_name(name: str) -> str:
    """Case/whitespace-insensitive key for matching a model-echoed test
    name back to the source passages we retrieved for it. Needed because
    not every model echoes the test name back byte-for-byte (Gemini
    happened to; Groq's models sometimes reword casing/spacing slightly),
    and a silent exact-match miss here means a finding loses its citation
    without any error being raised -- worse than a loud failure."""
    return re.sub(r"\s+", " ", name).strip().lower()


def _find_sources(test_name: str, sources_by_test: dict) -> list:
    """
    Look up which retrieved sources correspond to a model-echoed test
    name. Tries an exact normalized match first (handles case/whitespace
    differences); if that misses, falls back to substring containment in
    either direction, since some models expand abbreviations when they
    echo the test name back (e.g. input "WBC" comes back as "White Blood
    Cell Count (WBC)"). Returns [] if nothing matches either way, rather
    than guessing at an unrelated source.
    """
    key = _normalize_test_name(test_name)
    if key in sources_by_test:
        return sources_by_test[key]
    for stored_key, sources in sources_by_test.items():
        if stored_key in key or key in stored_key:
            return sources
    return []


def _find_scored_value(test_name: str, scored_by_test: dict) -> Optional[ScoredLabValue]:
    """
    Same fuzzy-match strategy as _find_sources() -- looks up the
    deterministically-scored value (see severity.py) that a model-echoed
    test name corresponds to. This is what status/severity/value/unit in
    the final Finding actually come from; the model's own copy of these
    fields in its JSON output is NEVER trusted for them, only used to
    figure out which test it's talking about and to write the free-text
    explanation. Silently trusting the model's echo defeats the entire
    point of computing severity deterministically (see severity.py's
    module docstring) -- if the model mis-copies a field, that error
    would otherwise reach the patient-facing report unnoticed.
    """
    key = _normalize_test_name(test_name)
    if key in scored_by_test:
        return scored_by_test[key]
    for stored_key, value in scored_by_test.items():
        if stored_key in key or key in stored_key:
            return value
    return None


def generate_report(scored_values: List[ScoredLabValue]) -> ReportResult:
    normal = [v for v in scored_values if v.severity == "normal"]
    unknown = [v for v in scored_values if v.severity == "unknown"]
    abnormal = [v for v in scored_values if v.severity in ("mild", "concerning")]

    if not abnormal:
        return ReportResult(
            overall_summary="All extracted results fall within their printed reference ranges.",
            findings=[],
            normal_findings=[v.test for v in normal],
            bottom_line=(
                "Nothing in this report falls outside the printed reference ranges. "
                "Still share these results with your doctor for full context."
            ),
            disclaimer=DISCLAIMER,
            insufficient_data=not scored_values,
            message="No test rows could be extracted from the file." if not scored_values else None,
        )

    # Retrieve grounding context per abnormal finding.
    context_blocks = []
    sources_by_test = {}
    scored_by_test = {}
    for v in abnormal:
        scored_by_test[_normalize_test_name(v.test)] = v
        chunks = retrieve_for_test(v.test)
        # Drop chunks that aren't actually relevant (see
        # RETRIEVAL_SCORE_THRESHOLD's comment in config.py) -- prevents
        # citing unrelated docs for a test with no real KB coverage.
        relevant_chunks = [c for c in chunks if c.score >= config.RETRIEVAL_SCORE_THRESHOLD]
        sources_by_test[_normalize_test_name(v.test)] = [c.source for c in relevant_chunks]
        chunk_text = (
            "\n\n".join(c.text for c in relevant_chunks)
            if relevant_chunks
            else "(no reference passage available in our knowledge base for this "
                 "specific test -- say so plainly rather than guessing)"
        )
        context_blocks.append(
            f"TEST: {v.test}\nVALUE: {v.value} {v.unit}\nREFERENCE RANGE: {v.reference_range}\n"
            f"COMPUTED STATUS: {v.status} ({v.severity}, {v.deviation_pct}% outside range)\n"
            f"REFERENCE PASSAGE:\n{chunk_text}"
        )

    user_prompt = (
        "Here are the abnormal findings, their computed status/severity, and "
        "the reference passage for each. Write the patient-facing report per "
        "the rules in your instructions.\n\n" + "\n\n---\n\n".join(context_blocks)
    )

    client = _get_client()
    try:
        completion = client.chat.completions.create(
            model=config.GROQ_TEXT_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": _SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": user_prompt},
            ],
        )
    except RateLimitError as e:
        # Groq's free tier has per-minute AND per-day quotas per model.
        # Both surface as this same exception type -- we can't tell which
        # one from the exception alone, so the message covers both
        # without guessing. This turns an opaque 500 + traceback into
        # something the frontend can actually show the user.
        raise GenerationRateLimitError(
            "The AI service's free-tier quota was reached (this can be a "
            "per-minute or per-day limit). Please wait a bit and try again, "
            "or check your Groq usage at https://console.groq.com/dashboard/limits."
        ) from e

    raw = _strip_code_fences(completion.choices[0].message.content or "{}")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON. Raw: {raw[:500]}") from e

    findings = []
    for f in parsed.get("findings", []):
        test_name = f.get("test", "")
        scored = _find_scored_value(test_name, scored_by_test)
        if scored is None:
            # Model referenced a test we never sent it -- skip rather
            # than fabricate a finding with no deterministic backing.
            continue
        findings.append(
            Finding(
                test=scored.test,
                value=str(scored.value),
                unit=scored.unit,
                status=scored.status,
                severity=scored.severity,
                plain_meaning=f.get("plain_meaning", ""),
                next_step=f.get("next_step", ""),
                sources=_find_sources(test_name, sources_by_test),
            )
        )

    return ReportResult(
        overall_summary=parsed.get("overall_summary", ""),
        findings=findings,
        normal_findings=[v.test for v in normal] + [v.test for v in unknown],
        bottom_line=parsed.get("bottom_line", ""),
        disclaimer=DISCLAIMER,
        insufficient_data=False,
    )