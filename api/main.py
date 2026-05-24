from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import load_and_split_pdf
from src.vectorstore import build_vectorstore, load_knowledge_base
from src.rag_chain import build_rag_chain, ask
from src.confidence import score_retrieval_confidence, should_fallback, build_fallback_response
from src.citation_verifier import verify_sources

app = FastAPI(
    title="RAG PDF Chatbot API",
    description="Production-grade RAG pipeline with hybrid search, reranking, and citation verification.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions = {}


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"
    mode: str = "personal"


class SourceDocument(BaseModel):
    source: str
    page: int
    content: str


class AskResponse(BaseModel):
    answer: str
    confidence: float
    is_fallback: bool
    verified_sources: List[SourceDocument]
    unverified_sources: List[SourceDocument]
    session_id: str


class IngestResponse(BaseModel):
    message: str
    chunks: int
    session_id: str


class DocumentInfo(BaseModel):
    name: str
    chunks: int
    session_id: str


def _doc_to_source(doc) -> SourceDocument:
    return SourceDocument(
        source=os.path.basename(doc.metadata.get("source", "unknown")),
        page=doc.metadata.get("page", 0),
        content=doc.page_content[:300]
    )


@app.get("/")
def root():
    return {
        "name": "RAG PDF Chatbot API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": ["/v1/ask", "/v1/ingest", "/v1/documents", "/v1/health"]
    }


@app.get("/v1/health")
def health():
    return {"status": "ok"}


@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Upload and index a PDF document.
    Returns a session_id to use in subsequent /v1/ask calls.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        chunks = load_and_split_pdf(tmp_path)
        os.unlink(tmp_path)

        vectorstore = build_vectorstore(chunks)
        chain = build_rag_chain(vectorstore, chunks)

        session_id = file.filename.replace(" ", "_")
        _sessions[session_id] = {
            "vectorstore": vectorstore,
            "chunks": chunks,
            "chain": chain,
            "filename": file.filename
        }

        return IngestResponse(
            message=f"Successfully indexed {file.filename}",
            chunks=len(chunks),
            session_id=session_id
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/v1/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Ask a question about an indexed document or the knowledge base.

    For personal document mode: provide session_id from /v1/ingest
    For knowledge base mode: set mode="kb"
    """
    if request.mode == "kb":
        vectorstore = load_knowledge_base()
        if vectorstore is None:
            raise HTTPException(
                status_code=404,
                detail="Knowledge base not found. Run scripts/seed_knowledge_base.py first."
            )
        kb_data = vectorstore.get()
        from langchain_core.documents import Document
        chunks = [
            Document(page_content=pc, metadata=meta)
            for pc, meta in zip(kb_data["documents"], kb_data["metadatas"])
        ]
        if request.session_id not in _sessions or _sessions[request.session_id].get("mode") != "kb":
            chain = build_rag_chain(vectorstore, chunks)
            _sessions[request.session_id] = {
                "vectorstore": vectorstore,
                "chunks": chunks,
                "chain": chain,
                "mode": "kb"
            }
        session = _sessions[request.session_id]

    else:
        if request.session_id not in _sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{request.session_id}' not found. Upload a document via /v1/ingest first."
            )
        session = _sessions[request.session_id]

    vectorstore = session["vectorstore"]
    chain = session["chain"]

    confidence = score_retrieval_confidence(vectorstore, request.question)

    if should_fallback(confidence):
        fallback = build_fallback_response(request.question, confidence)
        return AskResponse(
            answer=fallback["answer"],
            confidence=confidence,
            is_fallback=True,
            verified_sources=[],
            unverified_sources=[],
            session_id=request.session_id
        )

    result = ask(chain, request.question, session_id=request.session_id)
    verification = verify_sources(result["answer"], result["source_documents"])

    return AskResponse(
        answer=result["answer"],
        confidence=confidence,
        is_fallback=False,
        verified_sources=[_doc_to_source(d) for d in verification["verified_sources"]],
        unverified_sources=[_doc_to_source(d) for d in verification["unverified_sources"]],
        session_id=request.session_id
    )


@app.get("/v1/documents", response_model=List[DocumentInfo])
def list_documents():
    """
    List all currently indexed documents in active sessions.
    """
    docs = []
    for session_id, session in _sessions.items():
        docs.append(DocumentInfo(
            name=session.get("filename", session_id),
            chunks=len(session.get("chunks", [])),
            session_id=session_id
        ))
    return docs