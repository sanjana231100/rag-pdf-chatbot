from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import tempfile
import os
CHROMA_DIR = os.path.join(tempfile.gettempdir(), "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedder():
    """
    Load the HuggingFace embedding model.
    First run downloads ~80MB model to local cache — happens once only.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_vectorstore(chunks):
    """
    Take a list of Document chunks, embed them, and store in ChromaDB.

    Args:
        chunks: list of Document objects from ingest.py

    Returns:
        a Chroma vectorstore object ready for similarity search
    """
    embedder = get_embedder()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=CHROMA_DIR
    )

    return vectorstore


def load_vectorstore():
    """
    Load an existing ChromaDB from disk (if already built).
    Use this to avoid re-embedding on every app restart.

    Returns:
        a Chroma vectorstore object
    """
    embedder = get_embedder()

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedder
    )

    return vectorstore


def search(vectorstore, query, k=4):
    """
    Run a similarity search and return the top-k most relevant chunks.

    Args:
        vectorstore : the Chroma vectorstore object
        query       : the user's question as a plain string
        k           : number of chunks to retrieve (default 4)

    Returns:
        list of Document objects (each has .page_content and .metadata)
    """
    results = vectorstore.similarity_search(query, k=k)
    return results