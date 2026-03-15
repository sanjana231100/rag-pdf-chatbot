from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os


def load_and_split_pdf(file_source):
    """
    Load a PDF and split it into chunks.

    Args:
        file_source: either a file path string (for test scripts)
                     or a Streamlit UploadedFile object (for the app)

    Returns:
        list of Document objects, each with page_content and metadata
    """

    if hasattr(file_source, "read"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_source.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        os.unlink(tmp_path)
    else:
        loader = PyPDFLoader(file_source)
        pages = loader.load()

    if not pages or len(pages[0].page_content.strip()) < 50:
        raise ValueError(
            "Could not extract text from this PDF. "
            "It may be a scanned document — only text-based PDFs are supported."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1

    return chunks