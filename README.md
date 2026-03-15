# RAG PDF Chatbot

A conversational AI app that lets you upload PDF documents and ask questions about them in natural language. Built with LangChain, ChromaDB, and Streamlit — the same architecture used in enterprise tools like Microsoft Copilot and SAP Joule.

---

## What it does

1. Upload any PDF (HR policy, technical docs, contracts, etc.)
2. Ask questions in plain English
3. Get grounded answers with source citations (filename + page number)
4. Ask follow-up questions — conversation memory is built in

---

## Architecture

```
PDF → Text → Chunks → Embeddings → ChromaDB
                                        ↓
User Question → Embedding → Similar Chunks Retrieved
                                        ↓
               LLM (Llama 3 via Groq) reads chunks + question → Answer + Sources
```

---

## Tech stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| PDF parsing | PyPDFLoader (LangChain) |
| Text splitting | RecursiveCharacterTextSplitter |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace, free, local) |
| Vector store | ChromaDB |
| LLM | Llama 3 8B via Groq API (free) |
| Orchestration | LangChain |

---

## Run locally

**1. Clone the repo**
```bash
git clone https://github.com/your-username/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your Groq API key
# Get one free at https://console.groq.com
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## Project structure

```
rag-pdf-chatbot/
├── app.py              # Streamlit UI
├── requirements.txt
├── packages.txt        # HuggingFace Spaces system deps
├── README.md
├── .env.example
├── .gitignore
├── src/
│   ├── ingest.py       # PDF loading + chunking
│   ├── vectorstore.py  # ChromaDB build + load
│   └── rag_chain.py    # Retrieval chain + memory
└── data/               # Place test PDFs here
```

---

## Deployment

Deployed on [Hugging Face Spaces](https://huggingface.co/spaces).

Add your `GROQ_API_KEY` under **Settings → Repository Secrets** in your Space.

---

## Resume bullet

> Built a RAG pipeline ingesting PDF documents into a ChromaDB vector store, retrieving context-relevant chunks via semantic search, and generating grounded answers using Llama 3 via LangChain. Deployed on Hugging Face Spaces.