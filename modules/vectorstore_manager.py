# modules/vectorstore_manager.py
import os
import shutil
import logging
from typing import List, Optional

# LangChain loaders / utilities
from langchain.document_loaders import PDFPlumberLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# LangChain modular packages
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _load_pdf_safe(path: str, splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """
    Load and split PDF robustly: try PDFPlumber first, fallback to PyPDFLoader.
    Returns list[Document] (might be empty on failure).
    """
    try:
        loader = PDFPlumberLoader(path)
        docs = loader.load_and_split(text_splitter=splitter)
        logger.info("Loaded PDF with PDFPlumberLoader: %s (chunks=%d)", path, len(docs))
        return docs
    except Exception as e:
        logger.warning("PDFPlumberLoader failed for %s: %s — falling back to PyPDFLoader", path, e)
        try:
            loader = PyPDFLoader(path)
            docs = loader.load_and_split(text_splitter=splitter)
            logger.info("Loaded PDF with PyPDFLoader: %s (chunks=%d)", path, len(docs))
            return docs
        except Exception as e2:
            logger.error("Both PDF loaders failed for %s: %s", path, e2)
            return []


def persist_documents_to_chroma(
    uploaded_files,
    persist_directory: Optional[str] = None,
    collection_name: str = "kb_collection",
    overwrite: bool = True,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    embedding_model: str = "multi-qa-MiniLM-L6-cos-v1",
):
    """
    Build (or rebuild) a Chroma vectorstore from uploaded PDF files.

    - By default this creates an **in-memory** Chroma (suitable for Streamlit Cloud).
    - If `persist_directory` is provided, it will try to create a persistent collection there (for local dev).
    - If overwrite=True and persist_directory given, existing directory will be removed first.
    - Returns the Chroma vectorstore instance.
    """

    # prepare temp dir to save uploaded files (Streamlit ephemeral storage)
    tmp_dir = os.path.join("data", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # If persist_directory provided and overwrite requested, remove it to ensure clean rebuild
    if persist_directory:
        os.makedirs(os.path.dirname(persist_directory) or ".", exist_ok=True)
        if overwrite and os.path.exists(persist_directory):
            try:
                shutil.rmtree(persist_directory)
                logger.info("Removed existing vectorstore directory: %s", persist_directory)
            except Exception as e:
                logger.warning("Failed to remove vectorstore directory %s: %s", persist_directory, e)
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_docs: List[Document] = []

    # Save uploaded PDFs temporarily and extract docs
    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "name", "uploaded.pdf")
        temp_path = os.path.join(tmp_dir, filename)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        docs = _load_pdf_safe(temp_path, splitter)
        if not docs:
            logger.warning("No documents extracted from %s", filename)
        else:
            for d in docs:
                if d.metadata is None:
                    d.metadata = {}
                d.metadata["source"] = filename
            all_docs.extend(docs)

    if not all_docs:
        raise ValueError("No valid documents could be extracted from uploaded files.")

    # initialize embeddings
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    except Exception as e:
        logger.exception("Failed to initialize HuggingFaceEmbeddings: %s", e)
        raise

    # Build vectorstore: in-memory if persist_directory is None (safe for cloud)
    try:
        if persist_directory:
            vectordb = Chroma.from_documents(
                documents=all_docs,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
        else:
            # in-memory collection
            vectordb = Chroma.from_documents(
                documents=all_docs,
                embedding=embeddings,
                collection_name=collection_name,
            )
    except Exception as e:
        logger.exception("Failed to create Chroma vectorstore: %s", e)
        raise

    logger.info("Built vectorstore '%s' (docs=%d)", collection_name, len(all_docs))
    return vectordb
