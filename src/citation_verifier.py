from langchain_groq import ChatGroq
from langchain_core.documents import Document
from typing import List, Dict
from dotenv import load_dotenv
import os
import re

load_dotenv()

LLM_MODEL = "llama-3.3-70b-versatile"


def get_llm():
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )


def verify_citation(
    claim: str,
    source_chunk: str,
    llm
) -> bool:
    """
    Use an LLM to verify whether a source chunk actually supports a claim.

    Args:
        claim        : a single sentence or claim from the LLM's answer
        source_chunk : the text of the source chunk being cited
        llm          : the ChatGroq LLM instance

    Returns:
        True if the chunk supports the claim, False otherwise
    """
    prompt = f"""You are a fact-checking assistant. Your job is to determine whether 
a source passage supports a given claim.

Claim: {claim}

Source passage: {source_chunk}

Does the source passage directly support the claim? 
Answer with only YES or NO."""

    response = llm.invoke(prompt)
    answer = response.content.strip().upper()
    return answer.startswith("YES")


def extract_claims(answer: str) -> List[str]:
    """
    Split an answer into individual claims/sentences for verification.

    Args:
        answer : the LLM's full answer string

    Returns:
        list of individual claim strings
    """
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    claims = [s.strip() for s in sentences if len(s.strip()) > 20]
    return claims


def verify_sources(
    answer: str,
    source_documents: List[Document],
    max_claims: int = 3
) -> Dict:
    """
    Verify which source documents actually support the answer.

    For each source document, checks if it supports at least one
    claim in the answer. Only returns verified sources.

    Args:
        answer           : the LLM's answer string
        source_documents : list of Document objects from the RAG chain
        max_claims       : max number of claims to verify (to limit API calls)

    Returns:
        dict with:
            "verified_sources"   : list of Documents that are genuinely cited
            "unverified_sources" : list of Documents that don't support the answer
            "verification_rate"  : float — % of sources that are verified
    """
    if not source_documents or not answer:
        return {
            "verified_sources": source_documents,
            "unverified_sources": [],
            "verification_rate": 1.0
        }

    llm = get_llm()
    claims = extract_claims(answer)[:max_claims]

    if not claims:
        return {
            "verified_sources": source_documents,
            "unverified_sources": [],
            "verification_rate": 1.0
        }

    verified = []
    unverified = []

    for doc in source_documents:
        is_supported = False
        for claim in claims:
            try:
                if verify_citation(claim, doc.page_content[:500], llm):
                    is_supported = True
                    break
            except Exception:
                is_supported = True
                break

        if is_supported:
            verified.append(doc)
        else:
            unverified.append(doc)

    total = len(source_documents)
    verification_rate = len(verified) / total if total > 0 else 1.0

    return {
        "verified_sources": verified,
        "unverified_sources": unverified,
        "verification_rate": round(verification_rate, 2)
    }