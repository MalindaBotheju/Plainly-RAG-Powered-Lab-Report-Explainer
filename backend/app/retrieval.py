"""
Retrieval step.

Uses BM25 (classic lexical search, via rank_bm25) over a small local
knowledge base of plain-English lab test explanations. This is the "R" in
RAG for this project: instead of letting the LLM explain a lab value from
memory, we retrieve the relevant reference passage first and ground the
generation step in it.

Why BM25 instead of embeddings here: no API cost, no model download,
fully deterministic, and for a small (~10 document) knowledge base keyed
on exact test names, lexical search is genuinely competitive with dense
retrieval -- a good example of picking the simplest tool that fits the
problem rather than defaulting to embeddings everywhere.

A natural stretch upgrade (documented in the README) is hybrid BM25 +
dense embeddings with reciprocal rank fusion once the knowledge base
grows beyond a couple hundred documents.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

from . import config


@dataclass
class RetrievedChunk:
    source: str
    text: str
    score: float


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class KnowledgeBase:
    def __init__(self, directory: Path):
        self.documents = []  # list of (source_name, full_text)
        for path in sorted(directory.glob("*.txt")):
            self.documents.append((path.stem, path.read_text(encoding="utf-8")))

        self._corpus_tokens = [_tokenize(text) for _, text in self.documents]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        if not self.documents:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            range(len(self.documents)), key=lambda i: scores[i], reverse=True
        )
        results = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                continue
            source, text = self.documents[i]
            results.append(RetrievedChunk(source=source, text=text, score=float(scores[i])))
        return results


_kb_instance = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase(config.KNOWLEDGE_BASE_DIR)
    return _kb_instance


def retrieve_for_test(test_name: str, top_k: int = None) -> List[RetrievedChunk]:
    kb = get_knowledge_base()
    return kb.search(test_name, top_k=top_k or config.TOP_K_CHUNKS)
