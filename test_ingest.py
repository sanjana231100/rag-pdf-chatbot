"""
Test script for the PDF ingestion pipeline.
Run from the project root:  python test_ingest.py

Place any PDF in the data/ folder and update PDF_PATH below.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingest import load_and_split_pdf

PDF_PATH = "data/sample.pdf"

def test_ingestion():
    print(f"\nLoading PDF: {PDF_PATH}")
    print("-" * 50)

    try:
        chunks = load_and_split_pdf(PDF_PATH)
    except ValueError as e:
        print(f"Error: {e}")
        return
    except FileNotFoundError:
        print(f"File not found: {PDF_PATH}")
        print("Place a PDF in the data/ folder and update PDF_PATH in this script.")
        return

    print(f"Total chunks created : {len(chunks)}")
    print(f"Chunk size target    : 500 chars, overlap 50")
    print()

    print("--- Chunk 1 ---")
    print(f"Content  : {chunks[0].page_content[:300]}")
    print(f"Metadata : {chunks[0].metadata}")
    print()

    print("--- Chunk 2 ---")
    print(f"Content  : {chunks[1].page_content[:300]}")
    print(f"Metadata : {chunks[1].metadata}")
    print()

    lengths = [len(c.page_content) for c in chunks]
    print(f"Chunk length stats:")
    print(f"  Min : {min(lengths)} chars")
    print(f"  Max : {max(lengths)} chars")
    print(f"  Avg : {sum(lengths) // len(lengths)} chars")
    print()
    print("Ingestion pipeline working correctly.")

if __name__ == "__main__":
    test_ingestion()