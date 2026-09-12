import csv
import io
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .extraction import extract_lab_values
from .generation import GenerationRateLimitError, generate_report
from .severity import score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lab_report_agent")

app = FastAPI(
    title="Lab Report Explainer API",
    description="Upload a lab report (PDF/PNG/JPG) and get a plain-English, "
    "RAG-grounded explanation of the results.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=False,  # frontend sends no cookies/auth headers --
    # keeping this False lets ALLOWED_ORIGINS="*" work during early
    # testing without hitting the browser-enforced rule that wildcard
    # origins can't be combined with credentials.
    allow_methods=["*"],
    allow_headers=["*"],
)

AUDIT_LOG_PATH = Path(__file__).resolve().parent.parent / "audit_log.csv"


def _write_audit_row(filename: str, mime_type: str, num_findings: int, status: str):
    """
    Append-only local audit trail: every analyze call is logged with what
    happened, no patient-identifying info. On Render's ephemeral
    filesystem this resets on redeploy -- fine for a portfolio demo; a
    real deployment would write this to a persistent store instead.
    """
    is_new = not AUDIT_LOG_PATH.exists()
    with open(AUDIT_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp_utc", "filename", "mime_type", "num_findings", "status"])
        writer.writerow(
            [datetime.now(timezone.utc).isoformat(), filename, mime_type, num_findings, status]
        )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "vision_model": config.GROQ_VISION_MODEL,
        "text_model": config.GROQ_TEXT_MODEL,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in config.ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. "
            f"Allowed types: PDF, PNG, JPG/JPEG.",
        )

    raw_bytes = await file.read()
    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Max is {config.MAX_UPLOAD_MB}MB.",
        )

    try:
        lab_values = extract_lab_values(raw_bytes, file.content_type)
    except RuntimeError as e:
        # Missing API key etc -- a config problem, not a user error.
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        logger.warning("Extraction failed: %s", e)
        raise HTTPException(
            status_code=422,
            detail="Could not read test results from this file. Try a "
            "clearer scan/photo, or a different report.",
        )

    scored_values = [score(v) for v in lab_values]

    try:
        report = generate_report(scored_values)
    except GenerationRateLimitError as e:
        logger.warning("Gemini rate/quota limit hit: %s", e)
        raise HTTPException(status_code=429, detail=str(e))
    except ValueError as e:
        logger.warning("Generation failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Could not generate an explanation for these results. Please try again.",
        )

    _write_audit_row(
        filename=file.filename or "unknown",
        mime_type=file.content_type,
        num_findings=len(report.findings),
        status="ok",
    )

    result = asdict(report)
    return result
