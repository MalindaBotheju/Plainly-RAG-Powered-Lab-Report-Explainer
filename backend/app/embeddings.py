"""
Thin wrapper around the Jina AI Embeddings API (https://jina.ai/embeddings/).

Free tier: 100 requests/min, 100K tokens/min per API key, no credit card
required -- chosen for Step 2 of the RAG upgrade (retrieval.py stays as
the BM25 baseline; this is a second, independent retrieval method living
in embedding_retrieval.py).
"""
import socket
import time
from typing import List

import requests
import urllib3.util.connection as urllib3_cn

from . import config

JINA_URL = "https://api.jina.ai/v1/embeddings"

# Force IPv4-only DNS resolution for outbound connections. On some
# networks (VPNs, certain routers/ISPs) the machine has an IPv6 route
# that's a silent black hole -- no error, no refusal, the TCP connect()
# just hangs at the OS level for minutes, past any of the timeouts below,
# because the hang happens before Python's socket timeout logic can act.
# Symptom: request "works" once, then hangs forever on a later call with
# no exception, only fixable with Ctrl+C. Forcing IPv4 avoids ever trying
# the broken route.
def _allowed_gai_family():
    return socket.AF_INET


urllib3_cn.allowed_gai_family = _allowed_gai_family

# A handful of retries with backoff -- Jina's API sits behind Cloudflare,
# which can reset connections (BrokenPipeError/ConnectionError) on
# transient network hiccups or bot-detection false positives. A real 4xx
# (e.g. bad API key) is NOT retried, only connection-level failures.
_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 2


class JinaEmbeddingError(RuntimeError):
    pass


def embed_texts(texts: List[str], task: str) -> List[List[float]]:
    """
    task: "retrieval.passage" when embedding knowledge-base documents,
    "retrieval.query" when embedding a user query. Jina's models embed
    queries and passages asymmetrically for retrieval, so using the right
    task for each side matters for match quality.
    """
    if not config.JINA_API_KEY:
        raise JinaEmbeddingError(
            "JINA_API_KEY is not set. Get a free key at "
            "https://jina.ai/embeddings/ and set it in your .env (local) "
            "or Render environment variables (deployed)."
        )
    if not texts:
        return []

    last_error = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = requests.post(
                JINA_URL,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Connection": "close",
                    # A browser-like UA avoids some Cloudflare bot-detection
                    # edge cases that reset connections on the default
                    # "python-requests" UA.
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; lab-report-agent/1.0; "
                        "+https://github.com/)"
                    ),
                    "Authorization": f"Bearer {config.JINA_API_KEY}",
                },
                json={
                    "input": texts,
                    "model": config.JINA_EMBEDDING_MODEL,
                    "task": task,
                },
                timeout=(5, 15),  # (connect timeout, read timeout) seconds
            )
        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise JinaEmbeddingError(
                f"Connection to Jina API failed after {_MAX_RETRIES} attempts: {e}"
            ) from e

        if response.status_code != 200:
            raise JinaEmbeddingError(
                f"Jina API error {response.status_code}: {response.text[:300]}"
            )

        data = response.json()["data"]
        # The API doesn't guarantee response order matches input order --
        # each item carries its original "index", sort by that before
        # returning.
        ordered = sorted(data, key=lambda d: d["index"])
        return [d["embedding"] for d in ordered]

    # Unreachable, but keeps type-checkers happy.
    raise JinaEmbeddingError(str(last_error))
