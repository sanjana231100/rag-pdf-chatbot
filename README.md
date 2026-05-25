---
title: RAG PDF Chatbot
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# RAG PDF Chatbot

A production-grade Retrieval-Augmented Generation system with hybrid search, two-stage reranking, LLM-as-judge citation verification, and an automated evaluation framework.

**Live demo:** [huggingface.co/spaces/sanjana231100/rag-pdf-chatbot](https://huggingface.co/spaces/sanjana231100/rag-pdf-chatbot)

---

## The Problem

Standard RAG demos retrieve chunks using semantic search alone and pass them directly to an LLM. This fails in three ways: exact keyword queries (error codes, IDs, specific terms) get missed by embedding models, all retrieved chunks are treated as equally relevant without re-scoring, and the LLM may cite sources that don't actually support its claims.

This project addresses all three.

---

## Architecture

```
PDF / Knowledge Base
        ↓
   Text Chunks (500 chars, 50 overlap)
        ↓
  ┌─────────────────────────┐
  │   Hybrid Retrieval      │
  │  Semantic (MiniLM)  +   │  → top 20 candidates
  │  BM25 keyword search    │
  │  fused via RRF          │
  └─────────────────────────┘
        ↓
  Cross-encoder Reranker      → top 5 chunks
  (ms-marco-MiniLM-L-6-v2)
        ↓
  Confidence Scoring          → fallback if < 0.20
        ↓
  LLM (Llama 3.3 via Groq)   → grounded answer
        ↓
  Citation Verification       → LLM-as-judge per claim
  (LLM-as-judge)
        ↓
  Verified answer + sources
```

---

## Key Features

**Hybrid retrieval** — Dense semantic search (all-MiniLM-L6-v2) and BM25 sparse keyword search run in parallel. Results are fused using Reciprocal Rank Fusion. Semantic search finds conceptually similar chunks; BM25 catches exact keyword matches that embeddings miss.

**Two-stage reranking** — Hybrid search retrieves 20 candidates. A cross-encoder (ms-marco-MiniLM-L-6-v2) re-scores each (query, chunk) pair together, cutting to the top 5 most relevant. Cross-encoders are significantly more precise than bi-encoders because they see query and chunk jointly.

**Confidence-based fallback** — Retrieval confidence is scored before calling the LLM. If the score falls below threshold, a structured fallback response is returned instead of hallucinating. This is more useful than a fabricated answer.

**Citation verification** — After generation, each claim is sent to an LLM-as-judge to verify it's actually supported by the cited chunk. Only verified sources appear in the Sources panel.

**Dual mode** — Personal document assistant (user uploads any PDF) and a pre-indexed knowledge base mode (docs indexed once at setup, always available).

**FastAPI service layer** — REST API with `/v1/ask`, `/v1/ingest`, and `/v1/documents` endpoints. Auto-generated OpenAPI docs at `/docs`.

**Automated eval framework** — 50-question golden dataset with 5 difficulty categories. Automated metrics: answer correctness, faithfulness, retrieval relevance, and citation accuracy, all scored via LLM-as-judge.

---

## Eval Results

Evaluated on a 50-question golden dataset against a company HR policy corpus:

| Metric | Score |
|---|---|
| Answer correctness | ~88% (straightforward) |
| No-answer detection | 100% |
| Faithfulness | ~65% |
| Retrieval relevance | ~98% |

Difficulty breakdown:
- Straightforward lookups: ~88% correctness
- Multi-hop reasoning: ~70% correctness  
- No-answer questions: 100% correctly identified
- Ambiguous queries: ~50% correctness
- Misconception correction: ~67% correctness

---

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | all-MiniLM-L6-v2 (HuggingFace, free, local) |
| Sparse search | BM25 via rank-bm25 |
| Vector store | ChromaDB |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Llama 3.3 70B via Groq API (free) |
| Orchestration | LangChain v1.x LCEL |
| API | FastAPI + uvicorn |
| Frontend | Streamlit |
| Deployment | Docker / HuggingFace Spaces |

---

## Run Locally

**1. Clone and install:**
```bash
git clone https://github.com/sanjana231100/rag-pdf-chatbot.git
cd rag-pdf-chatbot
pip install -r requirements.txt
```

**2. Set up environment:**
```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
# Get one free at https://console.groq.com
```

**3. Seed the knowledge base (optional):**
```bash
# Add PDFs to data/knowledge_base/
python scripts/seed_knowledge_base.py
```

**4. Run the Streamlit app:**
```bash
streamlit run app.py
```

**5. Run the API (optional):**
```bash
uvicorn api.main:app --reload --port 8001
# Docs at http://localhost:8001/docs
```

**6. Run with Docker Compose:**
```bash
docker-compose up
```

**7. Run the eval framework:**
```bash
python eval/run_eval.py
```

---

## Project Structure

```
rag-pdf-chatbot/
├── app.py                      # Streamlit UI with dual mode
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── api/
│   └── main.py                 # FastAPI service layer
├── src/
│   ├── ingest.py               # PDF loading + chunking
│   ├── vectorstore.py          # ChromaDB build + load
│   ├── hybrid_retriever.py     # BM25 + semantic + RRF fusion
│   ├── reranker.py             # Cross-encoder reranker
│   ├── rag_chain.py            # Conversational RAG chain
│   ├── confidence.py           # Scoring + fallback logic
│   └── citation_verifier.py   # LLM-as-judge verification
├── eval/
│   ├── golden_qa.json          # 50-question dataset
│   └── run_eval.py             # Automated eval runner
├── scripts/
│   └── seed_knowledge_base.py  # Knowledge base indexing
└── data/
    └── knowledge_base/         # Pre-indexed docs
```

---

## Resume Bullet

> Engineered hybrid RAG pipeline combining dense vector search and BM25 sparse retrieval fused via Reciprocal Rank Fusion, with two-stage cross-encoder reranking and FastAPI service layer; supports multi-document ingestion and enterprise knowledge base mode. Built LLM-as-judge citation verification, confidence-threshold hallucination fallback, and automated eval framework (faithfulness, retrieval relevance, answer correctness) over 50-question golden dataset; deployed on HuggingFace Spaces with multi-turn memory.