# Plainly — RAG-Powered Lab Report Explainer

Plainly is a portfolio project that takes a photo of a lab report and explains the results in plain English, grounded in a small retrieval-augmented knowledge base rather than the model's own unverified medical "knowledge."

**Live demo:** `https://plainly-rag-powered-lab-report-expl.vercel.app`
**Not a medical device.** Always consult a licensed doctor. This tool exists to demonstrate RAG system design, not to give medical advice.

---

## 1. What this project actually demonstrates

Most "RAG chatbot" portfolio projects stop at "I added embeddings." This one is built around a different, more defensible claim: **every architectural decision here is backed by a measurement, not a guess**, including the times the "obviously better" approach lost.

Specifically:

- Retrieval quality was measured (Recall@K, MRR, Precision@K) across three methods — BM25, embeddings, and hybrid — with a hand-labeled test set split by difficulty (exact match / synonym / genuine paraphrase).
- The severity/status shown to the user is computed with **plain arithmetic**, never left to an LLM to judge — because LLM-judged severity is inconsistent across runs, and for a health-adjacent tool that inconsistency is a real risk, not a stylistic choice.
- Citations are enforced with fuzzy matching against real retrieved documents, not just asserted by the model.
- A real production bug was found and fixed where the *display* layer was silently trusting the LLM's own echoed copy of a deterministic field — see [§6](#6-a-real-bug-worth-highlighting-llm-echo-vs-deterministic-truth).

---

## 2. Architecture

```
Image (PNG/JPG)
      │
      ▼
Groq vision extraction  (qwen/qwen3.8-27b)
      │   → raw test rows: {test, value, unit, reference_range, reported_flag}
      ▼
Deterministic severity scoring  (severity.py — plain arithmetic, no LLM)
      │   → status: normal / low / high / unknown
      │   → severity: normal / mild / concerning / unknown
      ▼
Retrieval  (embedding_retrieval.py — Jina embeddings, chosen after evaluation)
      │   → top-3 relevant reference passages per abnormal test
      │   → passages below a similarity threshold are dropped, not cited
      ▼
Groq text generation  (openai/gpt-oss-20b)
      │   → plain-English explanation + next-step, grounded in retrieved passages only
      ▼
Grounded, cited report → frontend
```

**Backend:** FastAPI (Python), deployed on Render (free tier).
**Frontend:** static HTML/JS, deployed on Vercel.
**Retrieval knowledge base:** 12 short reference documents, one per lab test (`app/knowledge_base/*.txt`).

---

## 3. The retrieval story — how BM25, embeddings, and hybrid were actually compared

### Why this needed a real evaluation, not intuition

With a knowledge base this small (originally 10, now 12 documents), it would have been easy to just assume "embeddings are better, ship it." Instead, a labeled retrieval test set (`app/eval/retrieval_test_cases.json`) was built with 31 queries split into three difficulty tiers:

- **Exact** — the literal test name (e.g. "Glucose")
- **Synonym** — a common alternate phrasing (e.g. "Fasting Blood Sugar")
- **Paraphrase** — genuine semantic rewording with little to no shared vocabulary with the target document (e.g. "constantly exhausted, winded easily, looking washed out" → should retrieve the hemoglobin doc)

The paraphrase tier specifically was tuned to have close to zero literal word overlap with its target document — the first draft of this test set accidentally had BM25 scoring 100% even on "paraphrases," because the knowledge base documents happened to contain the same synonym words in their body text. The test set was rewritten to genuinely stress-test lexical vs. semantic matching before trusting any result from it.

### Results

| Method | Recall@3 | MRR | Precision@3 | Paraphrase recall |
|---|---|---|---|---|
| BM25 (lexical, original) | 90% | 0.89 | 57% | 75% |
| **Jina embeddings** | **100%** | **0.98** | 33% | **100%** |
| Hybrid (BM25 + embeddings, RRF) | 97% | 0.91 | 32% | 92% |

### The honest finding: hybrid didn't win

The expected narrative is "hybrid retrieval combines the best of both." That's not what happened here. **Hybrid retrieval scored *worse* than embeddings alone.** Reciprocal Rank Fusion blends *rankings*, not correctness — when BM25 confidently ranks a paraphrase query wrong (which it does, by design, since it has no semantic understanding), RRF still gives that wrong ranking equal weight, which can pull the correct answer down rather than reinforce it.

**Conclusion actually shipped:** pure Jina embeddings retrieval, used in production. A reranker (the next logical step in a "full RAG upgrade" checklist) was deliberately **not** added, because the measured evidence showed diminishing returns at this knowledge-base size — adding one anyway just to check a box would have been complexity without benefit.

Precision@3 dropping for embeddings (57% → 33%) is a side effect worth understanding, not a hidden regression: BM25 sometimes returns fewer than 3 candidates (zero-score documents are filtered out), which inflates its precision ratio arithmetically. Embeddings always fill all 3 slots via cosine similarity, even when only one is truly relevant.

---

## 4. Knowledge base coverage — a real, disclosed limitation

The knowledge base covers 12 specific lab tests:

`glucose`, `cholesterol_ldl`, `cholesterol_hdl`, `creatinine`, `hemoglobin`, `platelets`, `sodium_potassium`, `tsh`, `wbc`, `alt_ast`, `rbc`, `hct`

`rbc` and `hct` were added *after* live testing surfaced a real gap: a real CBC report has ~20 test types (MCV, MCH, MCHC, RDW, LYM%, ESR, etc.), and only a handful were originally covered. When an abnormal test has no matching reference document, the system does **not** silently cite the nearest unrelated document (an early, real bug — see below) — it says plainly that this specific test isn't in its reference library yet and to ask a doctor directly.

This is enforced with `RETRIEVAL_SCORE_THRESHOLD` (`config.py`): retrieved passages below a cosine-similarity cutoff are dropped before being shown to the generation model, rather than being cited as if they were relevant.

**Known limitation:** most CBC test components beyond the 12 above are still uncovered. Expanding this is the most straightforward next improvement.

---

## 5. Deterministic severity — the design principle behind `severity.py`

Status and severity are **never** decided by the LLM. `severity.py` parses the printed reference range (handling `70-99`, `<100`, `>40`, etc.), compares the numeric value against it with plain arithmetic, and classifies:

- Inside range → `normal`
- Outside range, ≤20% past the boundary → `mild`
- Outside range, >20% past the boundary → `concerning`
- Range or value unparseable → `unknown` (stated honestly, never guessed)

This is deliberately boring. An LLM asked "how worried should this patient be" is inconsistent run-to-run — for a tool people might actually read health information from, that inconsistency is a real risk worth designing around, not a nice-to-have.

---

## 6. A real bug worth highlighting: LLM echo vs. deterministic truth

Midway through live testing, one finding displayed **`unknown ↓ MILD`** — a status that shouldn't exist (a down-arrow direction with an "unknown" label). Root cause: `Finding.status` and `Finding.severity` were being populated from `f.get("status")` / `f.get("severity")` — **the LLM's own copy of these fields in its JSON output** — not from the already-computed, deterministic `ScoredLabValue`.

Gemini had happened to echo these fields back correctly during earlier testing, which is exactly why this went unnoticed for a while — it's the kind of bug that silently works until it doesn't. When the project moved to Groq's models, the echo was no longer reliable, and the bug surfaced.

**Fix:** `Finding.status`, `.severity`, `.value`, and `.unit` are now looked up directly from the deterministically-scored value (matched to the model's response via a fuzzy test-name match), never trusted from the model's own restatement. The model is only trusted for the two fields that are legitimately its job to generate: `plain_meaning` and `next_step`.

The same fuzzy-matching approach fixed a related citation bug: models don't always echo test names back byte-for-byte (case differences, or abbreviation expansion like `"WBC"` → `"White Blood Cell Count (WBC)"`), which silently broke exact-string citation lookups. Both now use normalized + substring-fallback matching (`_normalize_test_name`, `_find_sources`, `_find_scored_value` in `generation.py`).

---

## 7. LLM provider: why Groq, not Gemini

The project originally used Gemini (`google-generativeai`). Two independent problems forced a switch:

1. **Free-tier quota was unworkable for iteration.** A newly-created account was capped at 5 requests/minute and, worse, **20 requests/day** on the assigned model — exhausted after normal testing, not heavy use.
2. **The `google-generativeai` SDK is fully deprecated.** Google's own client raises a `FutureWarning` directing migration to `google.genai`.

Switched to **Groq**, which offers a far more workable free tier (thousands of requests/day depending on model) and isn't riding a deprecated SDK. This required two different Groq models, since Groq's fast/cheap text models are text-only:

- **Extraction (vision):** `qwen/qwen3.8-27b` — reads the uploaded image directly.
- **Generation (text):** `openai/gpt-oss-20b` — writes the plain-English report from already-extracted, already-scored values (no image input needed here).

**Trade-off accepted:** PDF upload support was dropped. Gemini could read PDFs natively as multimodal input; Groq's vision model only accepts images. Adding PDF support back would require a PDF→image conversion step (`poppler`/`pdf2image`), which was deliberately deferred to avoid adding a system-level dependency to an already-fragile first deployment. PNG/JPG only, for now.

Both Groq quota errors and the earlier Gemini ones are caught explicitly (`GenerationRateLimitError`) and surfaced to the frontend as a clean `429` with a clear message, instead of an opaque `500` and a raw traceback.

**Groq model names change frequently.** Two model names picked from documentation (`llama-3.1-8b-instant`, `meta-llama/llama-4-scout-17b-16e-instruct`) turned out to have been deprecated since the docs were written. The reliable way to pick a model is to query the account's live list directly:

```bash
curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
```

---

## 8. Embeddings: Jina AI

- Model: `jina-embeddings-v3`
- Free tier: 100 requests/minute, 100K tokens/minute, no credit card required
- Uses Jina's **asymmetric retrieval mode** — documents are embedded with `task="retrieval.passage"`, queries with `task="retrieval.query"` — since the model embeds each side slightly differently for better matching.
- Document embeddings are computed once and **cached to disk** (`knowledge_base_embeddings_cache.json`), auto-invalidating via a content hash if the knowledge base changes. This avoids re-embedding 12 static documents on every process restart, which matters on a free-tier deploy that spins down on inactivity.

### A networking bug worth documenting: the IPv6 black hole

Early testing showed the embeddings call **hanging indefinitely** — not erroring, just sitting there for minutes until manually killed — on some networks but not others, with no consistent pattern. Root cause: an IPv6 route that silently dropped packets (no error, no refusal) on certain networks/VPNs. Python's `socket.connect()` can hang past any request-level timeout when the hang happens at the OS/kernel level before the timeout logic gets a chance to act.

**Fix:** force IPv4-only DNS resolution for outbound calls to the Jina API (`urllib3.util.connection.allowed_gai_family` monkey-patched in `embeddings.py`), plus a `Connection: close` header and retry/backoff for genuine transient Cloudflare-related resets.

---

## 9. Evaluation harness

`app/eval/run_eval.py` — runs with zero API cost by default:

```bash
python -m app.eval.run_eval                              # severity + BM25 retrieval only, free
python -m app.eval.run_eval --with-embeddings             # + Jina embeddings + hybrid RRF retrieval evals
python -m app.eval.run_eval --with-embeddings --with-generation   # + real Groq generation calls
```

- `eval_severity()` — checks the deterministic status/severity rule against hand-labeled expected outputs.
- `eval_retrieval(k, retrieve_fn, label)` — generic retrieval evaluator (Recall@K / MRR / Precision@K), reused across BM25, embeddings, and hybrid so all three are measured identically.
- `eval_embedding_retrieval()`, `eval_hybrid_retrieval()` — call the above with each retriever.
- `eval_generation()` — runs the full pipeline on real test cases, checks that every finding has a citation, and flags any banned next-step language (specific drug names/dosages are never allowed in `next_step`).

Test data lives in `app/eval/test_cases.json` (severity/generation scenarios) and `app/eval/retrieval_test_cases.json` (31 labeled retrieval queries).

---

## 10. Deployment

**Backend (Render, free tier):**

```yaml
# render.yaml
services:
  - type: web
    name: lab-report-agent-api
    runtime: python
    plan: free
    rootDir: backend
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Required environment variables (set as secrets in Render's dashboard, not committed):

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq API access (extraction + generation) |
| `JINA_API_KEY` | Jina embeddings API access |
| `GROQ_VISION_MODEL` | default `qwen/qwen3.8-27b` |
| `GROQ_TEXT_MODEL` | default `openai/gpt-oss-20b` |
| `JINA_EMBEDDING_MODEL` | default `jina-embeddings-v3` |
| `ALLOWED_ORIGINS` | your frontend's exact URL — **not** `*` in production |
| `MAX_UPLOAD_MB` | default `10` |
| `TOP_K_CHUNKS` | default `3` |
| `RETRIEVAL_SCORE_THRESHOLD` | default `0.3` |

**Frontend (Vercel):** static HTML/JS in `frontend/`.

### CORS: a debugging note worth keeping

Two separate CORS issues came up during deployment, both worth knowing if this happens again:

1. `render.yaml` originally hardcoded `ALLOWED_ORIGINS: "*"` as a literal `value`. Since Render re-reads `render.yaml` on every deploy, this **silently overwrote** manual dashboard changes on every push. Fixed by switching it to `sync: false`, the same pattern already used for API key secrets — this tells Render "don't manage this value, leave whatever's set in the dashboard alone."
2. The backend originally set `allow_credentials=True` on `CORSMiddleware` without actually needing it (the frontend sends no cookies/auth headers). Per the CORS spec, browsers reject a wildcard origin combined with credentials — so `ALLOWED_ORIGINS="*"` looked broken even when correctly configured. Fixed by setting `allow_credentials=False`, matching what the frontend actually does.

---

## 11. Known limitations (disclosed, not hidden)

- **Knowledge base coverage is narrow** (12 tests). Any test outside this set is reported honestly as "not in the reference library" rather than misexplained.
- **No PDF support** — dropped when switching to Groq's vision model. Would require adding a PDF→image conversion step.
- **Free-tier rate limits** on both Groq and Jina — fine for demo/portfolio traffic, not for sustained real usage.
- **Not a medical device.** No diagnosis is ever made; `next_step` is restricted to monitoring, repeat testing, or seeing a doctor — never a specific medication or dosage (enforced both by prompt and by an eval check for banned words).

---

## 12. Local setup

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `backend/.env`:

```
GROQ_API_KEY=your_key_here
JINA_API_KEY=your_key_here
```

Run the eval harness first to confirm everything works before touching the API:

```bash
python -m app.eval.run_eval --with-embeddings --with-generation
```

Run the server:

```bash
uvicorn app.main:app --reload
```