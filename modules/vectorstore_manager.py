import os
import shutil
import logging
from typing import List
from langchain.document_loaders import PDFPlumberLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _load_pdf_safe(path: str, splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """Load PDF with fallback strategy to handle complex documents."""
    try:
        loader = PDFPlumberLoader(path)
        docs = loader.load_and_split(text_splitter=splitter)
        logger.info(f"Loaded PDF with PDFPlumberLoader: {path} ({len(docs)} chunks)")
        return docs
    except Exception as e:
        logger.warning(f"PDFPlumber failed for {path}: {e}. Falling back to PyPDFLoader.")
        try:
            loader = PyPDFLoader(path)
            docs = loader.load_and_split(text_splitter=splitter)
            logger.info(f"Loaded PDF with PyPDFLoader: {path} ({len(docs)} chunks)")
            return docs
        except Exception as e2:
            logger.error(f"Failed to load {path}: {e2}")
            return []


def persist_documents_to_chroma(
    uploaded_files,
    persist_directory: str = "data/vectorstore",
    collection_name: str = "kb_collection",
    overwrite: bool = True,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> Chroma:
    """
    Process uploaded PDFs into embeddings stored in a Chroma vectorstore.
    Uses Gemini embeddings for compatibility with Streamlit Cloud.
    """
    os.makedirs(os.path.dirname(persist_directory) or ".", exist_ok=True)

    # Auto-rebuild vectorstore (safe and lightweight)
    if overwrite and os.path.exists(persist_directory):
        shutil.rmtree(persist_directory, ignore_errors=True)
        logger.info(f"Rebuilt vectorstore directory: {persist_directory}")

    os.makedirs(persist_directory, exist_ok=True)
    tmp_dir = os.path.join("data", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_docs: List[Document] = []

    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "name", "uploaded.pdf")
        temp_path = os.path.join(tmp_dir, filename)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        docs = _load_pdf_safe(temp_path, splitter)
        for d in docs:
            d.metadata["source"] = filename
        all_docs.extend(docs)

    if not all_docs:
        raise ValueError("No valid documents extracted from uploads.")

    # ✅ Use Gemini Embeddings
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY. Please set it in Streamlit Secrets.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-exp-03-07",
        google_api_key=api_key,
    )

    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    try:
        vectordb.persist()
    except Exception:
        pass

    logger.info(f"✅ Vectorstore '{collection_name}' built with {len(all_docs)} chunks.")
    return vectordb
