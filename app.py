import streamlit as st
import os
from src.ingest import load_and_split_pdf
from src.vectorstore import build_vectorstore
from src.rag_chain import build_rag_chain, ask, store
from src.confidence import score_retrieval_confidence, should_fallback, build_fallback_response

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Chatbot")
st.caption("Upload a PDF and ask questions about it in natural language.")


def render_sources(sources):
    if not sources:
        return
    seen = set()
    unique_sources = []
    for doc in sources:
        key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:50])
        if key not in seen:
            seen.add(key)
            unique_sources.append(doc)

    with st.expander(f"Sources ({len(unique_sources)} chunks used)"):
        for i, doc in enumerate(unique_sources):
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "?")
            st.markdown(f"**{source} — page {page}**")
            st.caption(doc.page_content[:350] + "...")
            if i < len(unique_sources) - 1:
                st.divider()


with st.sidebar:
    st.header("Upload your document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Text-based PDFs only. Scanned documents are not supported."
    )

    if uploaded_file is not None:
        if "processed_file" not in st.session_state or \
                st.session_state.processed_file != uploaded_file.name:

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

            if uploaded_file.name in store:
                del store[uploaded_file.name]

            st.success(f"Ready! {len(chunks)} chunks indexed.")

    if "chunks" in st.session_state:
        st.divider()
        st.metric("Chunks indexed", len(st.session_state.chunks))
        st.caption(f"File: {st.session_state.processed_file}")
        st.caption("🔍 Hybrid search: semantic + BM25")

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
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.stop()

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            source_index = i // 2
            if source_index < len(st.session_state.source_history):
                render_sources(st.session_state.source_history[source_index])

if question := st.chat_input("Ask something about your document..."):

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
            else:
                result = ask(
                    st.session_state.chain,
                    question,
                    session_id=st.session_state.processed_file
                )
                result["confidence"] = confidence

        answer = result["answer"]
        sources = result.get("source_documents", [])

        st.write(answer)

        confidence_val = result.get("confidence", confidence)
        confidence_color = "green" if confidence_val >= 0.75 else "orange" if confidence_val >= 0.60 else "red"
        st.caption(f":{confidence_color}[Retrieval confidence: {confidence_val:.0%}]")

        render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.source_history.append(sources)