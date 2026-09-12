"""
Central configuration. All settings come from environment variables so the
same code runs locally (.env file) and on Render (dashboard env vars).
"""
import os
from pathlib import Path

# Load .env in local dev (Render/production sets real env vars directly,
# so this is a no-op there if python-dotenv isn't relevant).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent

# --- LLM provider (Groq free tier -- no credit card, generous limits) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Vision model for extraction (reads the uploaded image directly). Groq
# deprecates/renames models often -- if this 404s again, run:
#   curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
# and pick a model with "image" in input_modalities.
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
# Text model for generation (writing the plain-English report from
# already-extracted values -- no image input needed here).
GROQ_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "openai/gpt-oss-20b")

# --- Upload constraints ---
# PDF was supported under Gemini (native multimodal PDF reading). Groq's
# vision model only accepts images, not raw PDFs -- adding PDF support
# back would mean converting PDF pages to images server-side first (a
# poppler/pdf2image dependency), which we're deliberately not adding yet
# to keep the Render deploy simple. PNG/JPG only for now.
ALLOWED_MIME_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

# --- Retrieval ---
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "3"))
# Below this cosine similarity, a retrieved chunk is treated as "not
# actually relevant" rather than cited as a source. Without this, a test
# with no matching knowledge-base doc (e.g. one we haven't written a
# reference file for yet) would still get the nearest-neighbor docs
# attached as if they were real citations -- misleading, since they're
# not actually about that test. Exact cutoff is a rough default; tune
# based on real score distributions if false negatives/positives show up.
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.3"))

# --- Embedding retrieval (Jina AI, free tier: 100 RPM / 100K TPM) ---
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v3")
# Document embeddings are cached here after first build so we don't re-hit
# the API on every process restart (matters on a free-tier deploy that
# spins down on inactivity). Cache auto-invalidates if the KB text changes.
EMBEDDING_CACHE_PATH = BASE_DIR / "knowledge_base_embeddings_cache.json"

# --- CORS ---
# Comma-separated list of allowed frontend origins (set this on Render to
# your Vercel URL, e.g. "https://lab-report-agent.vercel.app").
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
