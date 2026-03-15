import streamlit as st
import tempfile
import os
from src.ingest import load_and_split_pdf
from src.vectorstore import build_vectorstore
from src.rag_chain import build_rag_chain, ask

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="docs",
    layout="wide"
)

st.title("PDF Chatbot")
st.caption("Upload a PDF and ask questions about it in natural language.")

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

            with st.spinner("Building vector store (first run downloads model ~80MB)..."):
                vectorstore = build_vectorstore(st.session_state.chunks)
                st.session_state.vectorstore = vectorstore

            with st.spinner("Initialising RAG chain..."):
                chain = build_rag_chain(st.session_state.vectorstore)
                st.session_state.chain = chain

            st.session_state.processed_file = uploaded_file.name
            st.session_state.messages = []
            st.session_state.source_history = []

            st.success(f"Ready! {len(chunks)} chunks indexed.")
            st.info(f"File: {uploaded_file.name}")

    if "chunks" in st.session_state:
        st.divider()
        st.metric("Chunks indexed", len(st.session_state.chunks))
        st.metric("File", st.session_state.processed_file)

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.source_history = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "source_history" not in st.session_state:
    st.session_state.source_history = []

if "chain" not in st.session_state:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant" and i // 2 < len(st.session_state.source_history):
            sources = st.session_state.source_history[i // 2]
            if sources:
                with st.expander("Sources"):
                    for doc in sources:
                        source = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page", "?")
                        st.caption(f"{os.path.basename(source)} — page {page}")
                        st.write(doc.page_content[:300] + "...")
                        st.divider()

if question := st.chat_input("Ask something about your document..."):

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(
                st.session_state.chain,
                question,
                session_id=st.session_state.processed_file
            )

        answer = result["answer"]
        sources = result["source_documents"]

        st.write(answer)

        if sources:
            with st.expander("Sources"):
                for doc in sources:
                    source = doc.metadata.get("source", "unknown")
                    page = doc.metadata.get("page", "?")
                    st.caption(f"{os.path.basename(source)} — page {page}")
                    st.write(doc.page_content[:300] + "...")
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.source_history.append(sources)