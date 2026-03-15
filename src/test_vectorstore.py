"""
Test script for the vectorstore pipeline.
Run from project root:  python test_vectorstore.py

Make sure you have a PDF in data/ and update PDF_PATH below.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingest import load_and_split_pdf
from src.vectorstore import build_vectorstore, search

PDF_PATH = "data/sample.pdf"
TEST_QUERY = "what is the main topic of this document"


def test_vectorstore():
    print("\nStep 1: Loading and chunking PDF...")
    print("-" * 50)
    chunks = load_and_split_pdf(PDF_PATH)
    print(f"Chunks created: {len(chunks)}")

    print("\nStep 2: Building ChromaDB vectorstore...")
    print("(First run downloads the embedding model ~80MB — be patient)")
    print("-" * 50)
    vectorstore = build_vectorstore(chunks)
    print("Vectorstore built and saved to ./chroma_db")

    print(f"\nStep 3: Running similarity search...")
    print(f"Query: '{TEST_QUERY}'")
    print("-" * 50)
    results = search(vectorstore, TEST_QUERY, k=4)

    print(f"Top {len(results)} chunks retrieved:\n")
    for i, doc in enumerate(results):
        print(f"  Result {i+1}")
        print(f"  Source : {doc.metadata.get('source', 'unknown')}")
        print(f"  Page   : {doc.metadata.get('page', '?')}")
        print(f"  Text   : {doc.page_content[:200]}...")
        print()

    print("Vectorstore pipeline working correctly.")
    print("You should see relevant chunks above — not random text.")


if __name__ == "__main__":
    test_vectorstore()