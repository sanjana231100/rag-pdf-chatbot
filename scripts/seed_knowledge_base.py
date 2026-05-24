"""
Seed script for the knowledge base.
Run once from the project root to index all PDFs in data/knowledge_base/:

    python scripts/seed_knowledge_base.py

This creates a persistent ChromaDB collection named "knowledge_base"
that the app loads on startup in knowledge base mode.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import load_and_split_pdf
from src.vectorstore import get_embedder
from langchain_chroma import Chroma
import tempfile

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "knowledge_base")
CHROMA_DIR = os.path.join(tempfile.gettempdir(), "chroma_kb")
COLLECTION_NAME = "knowledge_base"


def seed():
    print(f"\nLooking for PDFs in: {KB_DIR}")
    print("-" * 50)

    if not os.path.exists(KB_DIR):
        print(f"Directory not found: {KB_DIR}")
        print("Create data/knowledge_base/ and add PDF files to it.")
        return

    pdf_files = [f for f in os.listdir(KB_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in data/knowledge_base/")
        print("Add PDF files to data/knowledge_base/ and run this script again.")
        return

    print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

    all_chunks = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(KB_DIR, pdf_file)
        print(f"\nProcessing: {pdf_file}")
        try:
            chunks = load_and_split_pdf(pdf_path)
            print(f"  → {len(chunks)} chunks")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  → Error: {e}")
            continue

    if not all_chunks:
        print("\nNo chunks created. Check your PDF files.")
        return

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("\nBuilding embeddings and indexing into ChromaDB...")
    print("(First run downloads the embedding model ~80MB)")
    print("-" * 50)

    embedder = get_embedder()

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedder,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    print(f"\nKnowledge base indexed successfully!")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Location   : {CHROMA_DIR}")
    print(f"Documents  : {len(pdf_files)}")
    print(f"Chunks     : {len(all_chunks)}")
    print(f"\nYou can now run the app and switch to Knowledge Base mode.")


if __name__ == "__main__":
    seed()