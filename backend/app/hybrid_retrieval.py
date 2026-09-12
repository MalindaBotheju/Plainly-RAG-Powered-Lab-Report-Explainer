"""
Hybrid retrieval -- Step 3 of the RAG upgrade.

Combines BM25 (retrieval.py) and Jina embeddings (embedding_retrieval.py)
using Reciprocal Rank Fusion (RRF): each retriever produces its own full
ranking of the knowledge base, and a document's hybrid score is the sum
of 1/(RRF_K + rank) across the retrievers it appears in.

RRF is used instead of a weighted sum of raw scores because BM25 scores
(unbounded, corpus-dependent) and cosine similarities (0-1) live on
incomparable scales -- normalizing them well is fiddly and fragile, while
RRF sidesteps the problem entirely by only ever looking at rank position,
not the underlying score. It's the standard choice for this kind of
two-retriever combination.
"""
from typing import List

from .retrieval import get_knowledge_base, RetrievedChunk
from .embedding_retrieval import get_embedding_knowledge_base

RRF_K = 60  # standard constant from the original RRF paper (Cormack et al.)


def search(query: str, top_k: int = 3) -> List[RetrievedChunk]:
    bm25_kb = get_knowledge_base()
    embed_kb = get_embedding_knowledge_base()

    n_docs = len(bm25_kb.documents)
    if n_docs == 0:
        return []

    # Pull each retriever's FULL ranking (not just top_k) so RRF has a
    # complete rank position for every document it did return, even ones
    # that wouldn't have made either individual retriever's top-3.
    bm25_ranked = bm25_kb.search(query, top_k=n_docs)
    embed_ranked = embed_kb.search(query, top_k=n_docs)

    rrf_scores = {}
    text_by_source = {}

    for rank, chunk in enumerate(bm25_ranked, start=1):
        rrf_scores[chunk.source] = rrf_scores.get(chunk.source, 0.0) + 1.0 / (RRF_K + rank)
        text_by_source[chunk.source] = chunk.text

    for rank, chunk in enumerate(embed_ranked, start=1):
        rrf_scores[chunk.source] = rrf_scores.get(chunk.source, 0.0) + 1.0 / (RRF_K + rank)
        text_by_source[chunk.source] = chunk.text

    ordered_sources = sorted(rrf_scores, key=lambda s: rrf_scores[s], reverse=True)

    return [
        RetrievedChunk(source=s, text=text_by_source[s], score=rrf_scores[s])
        for s in ordered_sources[:top_k]
    ]
