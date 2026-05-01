from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from typing import List

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None


def get_reranker() -> CrossEncoder:
    """
    Load the cross-encoder model.
    Cached globally so it only loads once per session.
    First run downloads ~85MB model to local cache.
    """
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(
    query: str,
    candidates: List[Document],
    top_k: int = 5
) -> List[Document]:
    """
    Re-score candidate chunks using a cross-encoder and return top_k.

    Unlike bi-encoders (which embed query and chunk separately),
    a cross-encoder sees the query and chunk together as a pair,
    making it significantly more accurate at judging relevance.

    Args:
        query      : the user's question
        candidates : list of Document objects from hybrid search (top 20)
        top_k      : number of chunks to keep after reranking (default 5)

    Returns:
        list of top_k most relevant Documents sorted by reranker score
    """
    if not candidates:
        return []

    reranker = get_reranker()

    pairs = [(query, doc.page_content) for doc in candidates]

    scores = reranker.predict(pairs)

    scored_docs = list(zip(scores, candidates))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    top_docs = [doc for _, doc in scored_docs[:top_k]]

    return top_docs