"""
Embedding-based retrieval -- Step 2 of the RAG upgrade.

Lives alongside retrieval.py (BM25) rather than replacing it. See
retrieval.py's module docstring for why BM25 was the original choice;
Step 3 combines both into hybrid retrieval.

Document embeddings are computed once and cached to disk, since the
knowledge base is small and static -- this avoids re-hitting the Jina API
on every process restart, which matters on a free-tier deploy that spins
down on inactivity. The cache auto-invalidates if any .txt file in the
knowledge base changes (content hash check).
"""
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from . import config
from .embeddings import embed_texts


@dataclass
class RetrievedChunk:
    source: str
    text: str
    score: float


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _fingerprint(documents: List[Tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for name, text in documents:
        h.update(name.encode("utf-8"))
        h.update(text.encode("utf-8"))
    return h.hexdigest()


class EmbeddingKnowledgeBase:
    def __init__(self, directory: Path):
        self.documents = []  # list of (source_name, full_text)
        for path in sorted(directory.glob("*.txt")):
            self.documents.append((path.stem, path.read_text(encoding="utf-8")))

        self._vectors = self._load_or_build_vectors()

    def _load_or_build_vectors(self) -> List[List[float]]:
        current_hash = _fingerprint(self.documents)
        cache_path = config.EMBEDDING_CACHE_PATH

        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            if (
                cached.get("hash") == current_hash
                and cached.get("model") == config.JINA_EMBEDDING_MODEL
            ):
                return cached["vectors"]

        texts = [text for _, text in self.documents]
        vectors = embed_texts(texts, task="retrieval.passage")

        cache_path.write_text(json.dumps({
            "hash": current_hash,
            "model": config.JINA_EMBEDDING_MODEL,
            "vectors": vectors,
        }))
        return vectors

    def search(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        if not self.documents:
            return []

        query_vec = embed_texts([query], task="retrieval.query")[0]

        scored = [
            (i, _cosine(query_vec, self._vectors[i]))
            for i in range(len(self.documents))
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)

        results = []
        for i, sim in scored[:top_k]:
            source, text = self.documents[i]
            results.append(RetrievedChunk(source=source, text=text, score=sim))
        return results


_kb_instance = None


def get_embedding_knowledge_base() -> EmbeddingKnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = EmbeddingKnowledgeBase(config.KNOWLEDGE_BASE_DIR)
    return _kb_instance


def retrieve_for_test(test_name: str, top_k: int = None) -> List[RetrievedChunk]:
    """
    Same signature as retrieval.retrieve_for_test() (BM25) -- this is the
    drop-in replacement used by generation.py. Embeddings were chosen over
    BM25 and over hybrid RRF based on the eval harness results (see
    eval/run_eval.py): 100% recall vs. BM25's 90%, and hybrid actually
    scored lower (97%) than embeddings alone on this knowledge base -- see
    hybrid_retrieval.py's docstring for why.
    """
    kb = get_embedding_knowledge_base()
    return kb.search(test_name, top_k=top_k or config.TOP_K_CHUNKS)
