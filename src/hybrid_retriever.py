from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from typing import List
import numpy as np


def build_bm25_index(chunks: List[Document]) -> BM25Okapi:
    """
    Build a BM25 index from a list of Document chunks.
    Tokenises each chunk by splitting on whitespace.

    Args:
        chunks: list of Document objects from ingest.py

    Returns:
        a BM25Okapi index ready for keyword search
    """
    tokenised = [chunk.page_content.lower().split() for chunk in chunks]
    return BM25Okapi(tokenised)


def bm25_search(
    bm25_index: BM25Okapi,
    chunks: List[Document],
    query: str,
    k: int = 20
) -> List[tuple]:
    """
    Run BM25 keyword search over the chunks.

    Args:
        bm25_index : the BM25Okapi index from build_bm25_index()
        chunks     : the original list of Document objects
        query      : the user's question
        k          : number of results to return

    Returns:
        list of (Document, bm25_score) tuples sorted by score descending
    """
    tokenised_query = query.lower().split()
    scores = bm25_index.get_scores(tokenised_query)

    top_k_indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_k_indices:
        if scores[idx] > 0:
            results.append((chunks[idx], float(scores[idx])))

    return results


def reciprocal_rank_fusion(
    semantic_results: List[Document],
    bm25_results: List[tuple],
    k: int = 60,
    top_n: int = 20
) -> List[Document]:
    """
    Combine semantic and BM25 results using Reciprocal Rank Fusion.

    RRF score = 1/(k + rank_in_semantic) + 1/(k + rank_in_bm25)
    Chunks appearing in both lists get a double boost.

    Args:
        semantic_results : list of Documents from ChromaDB similarity search
        bm25_results     : list of (Document, score) tuples from bm25_search()
        k                : RRF constant (60 is standard)
        top_n            : number of fused results to return

    Returns:
        list of Documents sorted by RRF score descending
    """
    rrf_scores = {}
    doc_map = {}

    for rank, doc in enumerate(semantic_results):
        key = doc.page_content[:100]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    for rank, (doc, _) in enumerate(bm25_results):
        key = doc.page_content[:100]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    return [doc_map[key] for key in sorted_keys[:top_n]]


def hybrid_search(
    vectorstore,
    bm25_index: BM25Okapi,
    chunks: List[Document],
    query: str,
    top_n: int = 20
) -> List[Document]:
    """
    Full hybrid search pipeline: semantic + BM25 + RRF fusion.

    Args:
        vectorstore : Chroma vectorstore object
        bm25_index  : BM25Okapi index
        chunks      : original list of Document chunks
        query       : user's question
        top_n       : number of candidates to return before reranking

    Returns:
        list of top_n Documents sorted by RRF score
    """
    semantic_results = vectorstore.similarity_search(query, k=top_n)

    bm25_results = bm25_search(bm25_index, chunks, query, k=top_n)

    fused_results = reciprocal_rank_fusion(
        semantic_results,
        bm25_results,
        top_n=top_n
    )

    return fused_results