from langchain_core.documents import Document
from typing import List, Dict


CONFIDENCE_THRESHOLD = 0.60


def score_retrieval_confidence(
    vectorstore,
    query: str,
    top_k: int = 5
) -> float:
    """
    Score how confident we are that the vectorstore contains
    a good answer to this query.

    Uses cosine similarity scores from ChromaDB's similarity_search_with_score.
    Returns the average of the top-k similarity scores, normalised to 0-1.

    Args:
        vectorstore : Chroma vectorstore object
        query       : the user's question
        top_k       : number of chunks to score against

    Returns:
        float between 0 and 1 — higher means more confident
    """
    try:
        results_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)

        if not results_with_scores:
            return 0.0

        scores = []
        for _, score in results_with_scores:
            normalised = max(0.0, min(1.0, 1.0 - score))
            scores.append(normalised)

        return round(sum(scores) / len(scores), 3)

    except Exception:
        return 0.5


def build_fallback_response(query: str, confidence: float) -> Dict:
    """
    Build a structured fallback response when confidence is too low.
    Returns this instead of calling the LLM to avoid hallucination.

    Args:
        query      : the user's original question
        confidence : the retrieval confidence score

    Returns:
        dict with "answer" and "source_documents" keys
        matching the format of a normal ask() response
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

    Args:
        confidence : retrieval confidence score from score_retrieval_confidence()

    Returns:
        True if confidence is below threshold and we should not call the LLM
    """
    return confidence < CONFIDENCE_THRESHOLD