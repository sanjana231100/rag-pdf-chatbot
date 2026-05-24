from langchain_core.documents import Document
from typing import List, Dict

CONFIDENCE_THRESHOLD = 0.20


def score_retrieval_confidence(
    vectorstore,
    query: str,
    top_k: int = 5
) -> float:
    """
    Score how confident we are that the vectorstore contains
    a good answer to this query.

    ChromaDB with cosine distance returns scores where:
    - 0.0 = identical (best match)
    - 2.0 = completely opposite (worst match)
    - Typical good matches score between 0.2 and 0.8

    We normalise to 0-1 where 1 = most confident.
    """
    try:
        results_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)

        if not results_with_scores:
            return 0.0

        scores = []
        for _, distance in results_with_scores:
            normalised = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
            scores.append(normalised)

        return round(sum(scores) / len(scores), 3)

    except Exception:
        return 0.5


def build_fallback_response(query: str, confidence: float) -> Dict:
    """
    Build a structured fallback response when confidence is too low.
    """
    answer = (
        f"I couldn't find relevant information about this in the uploaded document "
        f"(retrieval confidence: {confidence:.0%}). "
        f"The document may not contain information about '{query}'. "
        f"Try rephrasing your question or check if the document covers this topic."
    )

    return {
        "answer": answer,
        "source_documents": [],
        "confidence": confidence,
        "is_fallback": True
    }


def should_fallback(confidence: float) -> bool:
    """
    Decide whether to skip the LLM and return a fallback response.
    """
    return confidence < CONFIDENCE_THRESHOLD