import streamlit as st
import os
from src.ingest import load_and_split_pdf
from src.vectorstore import build_vectorstore, load_knowledge_base
from src.rag_chain import build_rag_chain, ask, store
from src.confidence import score_retrieval_confidence, should_fallback, build_fallback_response
from src.citation_verifier import verify_sources

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Chatbot")
st.caption("Upload a PDF and ask questions about it in natural language.")


def render_sources(sources, unverified=None):
    if not sources:
        return
    seen = set()
    unique_sources = []
    for doc in sources:
        key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:50])
        if key not in seen:
            seen.add(key)
            unique_sources.append(doc)

    with st.expander(f"✅ Verified sources ({len(unique_sources)} chunks)"):
        for i, doc in enumerate(unique_sources):
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "?")
            st.markdown(f"**{source} — page {page}**")
            st.caption(doc.page_content[:350] + "...")
            if i < len(unique_sources) - 1:
                st.divider()

    if unverified:
        with st.expander(f"⚠️ Unverified sources ({len(unverified)} chunks)"):
            st.caption("These chunks were retrieved but may not directly support the answer.")
            for doc in unverified:
                source = os.path.basename(doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page", "?")
                st.markdown(f"**{source} — page {page}**")
                st.caption(doc.page_content[:200] + "...")


with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "Select mode",
        ["📄 Personal document", "🏢 Knowledge base"],
        label_visibility="collapsed"
    )

    st.divider()

    if mode == "📄 Personal document":
        st.header("Upload your document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Text-based PDFs only. Scanned documents are not supported."
        )

        if uploaded_file is not None:
            if "processed_file" not in st.session_state or \
                    st.session_state.processed_file != uploaded_file.name or \
                    st.session_state.get("mode") != "personal":

                with st.spinner("Reading and chunking PDF..."):
                    try:
                        chunks = load_and_split_pdf(uploaded_file)
                        st.session_state.chunks = chunks
                    except ValueError as e:
                        st.error(str(e))
                        st.stop()

                with st.spinner("Building vector store and BM25 index..."):
                    vectorstore = build_vectorstore(st.session_state.chunks)
                    st.session_state.vectorstore = vectorstore

                with st.spinner("Initialising hybrid RAG chain..."):
                    chain = build_rag_chain(
                        st.session_state.vectorstore,
                        st.session_state.chunks
                    )
                    st.session_state.chain = chain

                st.session_state.processed_file = uploaded_file.name
                st.session_state.messages = []
                st.session_state.source_history = []
                st.session_state.mode = "personal"

                if uploaded_file.name in store:
                    del store[uploaded_file.name]

                st.success(f"Ready! {len(chunks)} chunks indexed.")

        if "chunks" in st.session_state and st.session_state.get("mode") == "personal":
            st.divider()
            st.metric("Chunks indexed", len(st.session_state.chunks))
            st.caption(f"File: {st.session_state.processed_file}")

    else:
        st.header("Knowledge base")

        if "kb_loaded" not in st.session_state or not st.session_state.kb_loaded or \
                st.session_state.get("mode") != "kb":

            with st.spinner("Loading knowledge base..."):
                kb_vectorstore = load_knowledge_base()

            if kb_vectorstore is None:
                st.error("Knowledge base not found.")
                st.info(
                    "Run the seed script first:\n\n"
                    "```\npython scripts/seed_knowledge_base.py\n```\n\n"
                    "Add PDFs to `data/knowledge_base/` before running."
                )
                st.stop()
            else:
                kb_chunks = kb_vectorstore.get()
                from langchain_core.documents import Document
                chunks = [
                    Document(page_content=pc, metadata=meta)
                    for pc, meta in zip(
                        kb_chunks["documents"],
                        kb_chunks["metadatas"]
                    )
                ]
                st.session_state.vectorstore = kb_vectorstore
                st.session_state.chunks = chunks
                st.session_state.chain = build_rag_chain(kb_vectorstore, chunks)
                st.session_state.processed_file = "knowledge_base"
                st.session_state.messages = []
                st.session_state.source_history = []
                st.session_state.kb_loaded = True
                st.session_state.mode = "kb"
                st.success(f"Knowledge base loaded — {len(chunks)} chunks ready.")

        if st.session_state.get("kb_loaded"):
            st.caption(f"Chunks: {len(st.session_state.chunks)}")

    st.divider()
    st.caption("🔍 Hybrid search: semantic + BM25")
    st.caption("🎯 Two-stage reranking: cross-encoder")
    st.caption("✅ Citation verification: LLM-as-judge")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.source_history = []
        if "processed_file" in st.session_state:
            if st.session_state.processed_file in store:
                del store[st.session_state.processed_file]
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "source_history" not in st.session_state:
    st.session_state.source_history = []

if "chain" not in st.session_state:
    if mode == "📄 Personal document":
        st.info("👈 Upload a PDF in the sidebar to get started.")
    else:
        st.info("👈 Loading knowledge base...")
    st.stop()

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            source_index = i // 2
            if source_index < len(st.session_state.source_history):
                entry = st.session_state.source_history[source_index]
                render_sources(
                    entry.get("verified", []),
                    entry.get("unverified", [])
                )

if question := st.chat_input("Ask something..."):

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            confidence = score_retrieval_confidence(
                st.session_state.vectorstore,
                question
            )

            if should_fallback(confidence):
                result = build_fallback_response(question, confidence)
                verified_sources = []
                unverified_sources = []
            else:
                result = ask(
                    st.session_state.chain,
                    question,
                    session_id=st.session_state.processed_file
                )
                result["confidence"] = confidence

                with st.spinner("Verifying citations..."):
                    verification = verify_sources(
                        result["answer"],
                        result["source_documents"]
                    )
                verified_sources = verification["verified_sources"]
                unverified_sources = verification["unverified_sources"]

        answer = result["answer"]
        confidence_val = result.get("confidence", confidence)

        st.write(answer)

        confidence_color = "green" if confidence_val >= 0.75 else "orange" if confidence_val >= 0.60 else "red"
        st.caption(f":{confidence_color}[Retrieval confidence: {confidence_val:.0%}]")

        render_sources(verified_sources, unverified_sources)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.source_history.append({
        "verified": verified_sources,
        "unverified": unverified_sources
    })