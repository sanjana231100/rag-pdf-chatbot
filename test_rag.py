"""
End-to-end test of the full RAG pipeline.
Run from project root:  python test_rag.py

Requires:
  - A PDF in data/ (update PDF_PATH below)
  - GROQ_API_KEY set in your .env file
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingest import load_and_split_pdf
from src.vectorstore import build_vectorstore
from src.rag_chain import build_rag_chain, ask

PDF_PATH = "data/sample.pdf"

TEST_QUESTIONS = [
    "What is this document about?",
    "Can you summarise the key points?",
    "What does it say about policies or rules?",
]


def test_rag():
    print("\nStep 1: Ingesting PDF...")
    print("-" * 50)
    chunks = load_and_split_pdf(PDF_PATH)
    print(f"Chunks: {len(chunks)}")

    print("\nStep 2: Building vectorstore...")
    print("-" * 50)
    vectorstore = build_vectorstore(chunks)
    print("Vectorstore ready")

    print("\nStep 3: Building RAG chain...")
    print("-" * 50)
    chain = build_rag_chain(vectorstore)
    print("Chain ready")

    print("\nStep 4: Asking questions...")
    print("=" * 50)

    for question in TEST_QUESTIONS:
        print(f"\nQ: {question}")
        print("-" * 50)

        result = ask(chain, question)

        print(f"A: {result['answer']}")
        print(f"\nSources used ({len(result['source_documents'])} chunks):")

        for doc in result["source_documents"]:
            print(f"  - {doc.metadata.get('source')} | page {doc.metadata.get('page')}")
            print(f"    \"{doc.page_content[:120]}...\"")

        print()

    print("=" * 50)
    print("Full RAG pipeline working correctly.")
    print("If answers look grounded in the document, you are ready for commit 5.")


if __name__ == "__main__":
    test_rag()