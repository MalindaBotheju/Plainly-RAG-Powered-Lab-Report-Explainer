# Plainly — RAG-Powered Lab Report Explainer

Upload a lab report (PDF, PNG, or JPG) and get a short, plain-English
explanation of the results: what's off, what it can mean, and what to ask
your doctor next — grounded in a retrieved reference passage for every
abnormal value, not generated from the model's memory alone.

**This is a portfolio/demo project, not a medical device.** It is
explicitly designed to never diagnose or recommend specific treatment —
see "Safety design" below.

---

## Why this project (the RAG story)

Most "chat with your PDF" RAG demos retrieve generic chunks and hope for
the best. This project narrows the scope so retrieval actually matters:

- **Deterministic severity, not LLM judgment.** Whether a value is
  `normal / mild / concerning` is computed with plain arithmetic
  (`app/severity.py`) from the reference range printed on the report. The
  LLM never decides this — it's handed the answer. This removes a whole
  class of inconsistency/hallucination risk.
- **Retrieval grounds every explanation.** For each abnormal finding, the
  system retrieves the matching reference passage from a small
  hand-written knowledge base (`app/knowledge_base/*.txt`) using BM25
  lexical search, and the generation prompt requires the explanation to
  come from that passage — not general knowledge.
- **A constrained safety boundary, enforced in the prompt.** "Next step"
  suggestions are limited to *monitor / repeat test / see a doctor* —
  never a medication, dose, or specific treatment. The eval harness
  checks for this directly (see below).
- **An eval harness that actually measures something.** `app/eval/`
  contains hand-labeled test cases and a script that checks severity-rule
  accuracy (free, deterministic) and, optionally, whether generated
  findings cite sources and avoid banned treatment language.

---

## Architecture

```
 PDF / PNG / JPG
       │
       ▼
┌─────────────────┐   Gemini multimodal call, strict JSON out
│   extraction.py │   "read what's on the page, don't interpret it"
└────────┬─────────┘
         ▼
┌─────────────────┐   pure arithmetic, no LLM
│   severity.py   │   status (high/low/normal) + severity (mild/concerning)
└────────┬─────────┘
         ▼
┌─────────────────┐   BM25 over app/knowledge_base/*.txt
│  retrieval.py   │   one retrieval per abnormal finding
└────────┬─────────┘
         ▼
┌─────────────────┐   Gemini text call, grounded in retrieved passages,
│  generation.py  │   constrained by system prompt (plain English,
└────────┬─────────┘   bullets, no treatment advice)
         ▼
   JSON report  ──────►  frontend/index.html renders it
```

Every `/api/analyze` call is appended to `backend/audit_log.csv`
(timestamp, filename, mime type, number of findings, status) — a minimal
stand-in for the access-logging you'd want in any real health-adjacent
system.

---

## Project structure

```
lab-report-agent/
├── backend/
│   ├── app/
│   │   ├── main.py            FastAPI app, /api/analyze endpoint
│   │   ├── extraction.py      Gemini multimodal -> structured rows
│   │   ├── severity.py        deterministic status/severity rule
│   │   ├── retrieval.py       BM25 retrieval over knowledge_base/
│   │   ├── generation.py      grounded plain-English report generation
│   │   ├── schemas.py         shared dataclasses
│   │   ├── config.py          env var driven settings
│   │   ├── knowledge_base/    10 plain-English reference docs
│   │   └── eval/
│   │       ├── test_cases.json
│   │       └── run_eval.py
│   ├── requirements.txt
│   ├── Procfile                for Render
│   ├── render.yaml              Render deploy blueprint
│   └── .env.example
└── frontend/
    ├── index.html               single-file UI (HTML/CSS/JS, no build step)
    ├── config.js                one line: your backend URL
    └── vercel.json
```

---

## Local setup

### 1. Get a free Gemini API key
Go to <https://aistudio.google.com/apikey> and create a free key
(Gemini's free tier includes multimodal calls, which is why this project
uses it instead of a paid API).

### 2. Backend
```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your GEMINI_API_KEY

uvicorn app.main:app --reload --port 8000
```
Check it's alive: `curl http://localhost:8000/api/health`

### 3. Frontend
`frontend/config.js` already points at `http://localhost:8000` by default.
Just open `frontend/index.html` in a browser (or serve it: `python3 -m
http.server 5500` from inside `frontend/`).

Lets move ahead. Next steps are;

### 4. Run the eval harness
```bash
cd backend
python3 -m app.eval.run_eval                    # free, deterministic severity check
python3 -m app.eval.run_eval --with-generation   # also exercises the real Gemini calls
```

---

## Deployment

### Backend → Render
1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, point it at the repo — `backend/render.yaml`
   is picked up automatically (root dir is set to `backend`).
3. Set the `GEMINI_API_KEY` env var in the Render dashboard (marked
   `sync: false` in the blueprint, so it's not committed to git).
4. Once deployed, note your API URL, e.g.
   `https://lab-report-agent-api.onrender.com`.
5. Update `ALLOWED_ORIGINS` on Render to your Vercel URL once you have it
   (step below), so CORS only allows your actual frontend.

> Free-tier Render services spin down when idle — the first request after
> a while can take ~30-50 seconds to wake up. Worth mentioning in a demo
> so it doesn't look broken.

### Frontend → Vercel
1. In Vercel: **New Project**, import the same repo, set **Root
   Directory** to `frontend`. No build step needed (static HTML).
2. After deploy, edit `frontend/config.js`:
   ```js
   window.PLAINLY_API_BASE_URL = "https://lab-report-agent-api.onrender.com";
   ```
3. Commit and push — Vercel redeploys automatically.

---

## Safety design (worth calling out in interviews)

- Severity is rule-based, not LLM-judged (see above).
- The generation system prompt hard-bans specific medications, dosages,
  and treatment instructions in "next step" text — only *monitor / repeat
  test / see a doctor* are allowed. `eval/run_eval.py --with-generation`
  checks generated output for banned words as a regression guard.
- Every explanation must be grounded in a retrieved reference passage;
  the prompt explicitly tells the model not to fill gaps from general
  knowledge.
- The UI and API response both carry an explicit non-diagnostic
  disclaimer.
- No patient-identifying information is requested, stored, or logged —
  the audit log only records filename/mime-type/finding-count, not file
  contents.

## Known limitations (also worth calling out)

- Knowledge base covers ~10 common tests; an unrecognized test falls back
  to no retrieved context and is reported honestly as such, not
  hallucinated.
- Severity thresholds (±20% from the reference boundary) are a
  simplification — real clinical risk depends on more than distance from
  a range.
- Free-tier Gemini has rate limits; heavy demo traffic may hit them.
- BM25 retrieval is lexical, not semantic — a natural next step is hybrid
  BM25 + embedding retrieval once the knowledge base grows.
