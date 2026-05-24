from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile
import os

CHROMA_DIR = os.path.join(tempfile.gettempdir(), "chroma_db")
KB_CHROMA_DIR = os.path.join(tempfile.gettempdir(), "chroma_kb")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
KB_COLLECTION = "knowledge_base"


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
    Build a ChromaDB vectorstore from a list of Document chunks.
    Used for personal document mode (user uploads a PDF).
    """
    embedder = get_embedder()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=CHROMA_DIR
    )
    return vectorstore


def load_knowledge_base():
    """
    Load the pre-indexed knowledge base ChromaDB collection.
    Returns None if the knowledge base hasn't been seeded yet.
    """
    if not os.path.exists(KB_CHROMA_DIR):
        return None

    try:
        embedder = get_embedder()
        vectorstore = Chroma(
            collection_name=KB_COLLECTION,
            embedding_function=embedder,
            persist_directory=KB_CHROMA_DIR
        )
        count = vectorstore._collection.count()
        if count == 0:
            return None
        return vectorstore
    except Exception:
        return None


def search(vectorstore, query, k=4):
    """
    Run a similarity search and return the top-k most relevant chunks.
    """
    return vectorstore.similarity_search(query, k=k)