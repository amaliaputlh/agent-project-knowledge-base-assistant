# modules/vectorstore_manager.py
import os
import shutil
import logging
from typing import List, Optional

from langchain.document_loaders import PDFPlumberLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# New modular packages (avoid deprecation warnings)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _load_pdf_safe(path: str, splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """
    Try PDFPlumberLoader first (more robust). If it fails, fall back to PyPDFLoader.
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
            logger.error("Both loaders failed for %s: %s", path, e2)
            return []


def persist_documents_to_chroma(
    uploaded_files,
    persist_directory: str = "data/vectorstore",
    collection_name: str = "kb_collection",
    overwrite: bool = True,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    embedding_model: str = "multi-qa-MiniLM-L6-cos-v1",
) -> Chroma:
    """
    Convert uploaded Streamlit files into LangChain Document chunks,
    embed them using a HuggingFace embedding model, and save to Chroma vectorstore.

    - overwrite=True will remove existing persist_directory and rebuild (automatic rebuild).
    - returns the Chroma vectorstore instance.
    """

    os.makedirs(os.path.dirname(persist_directory) or ".", exist_ok=True)

    if overwrite and os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
            logger.info("Removed existing vectorstore directory: %s", persist_directory)
        except Exception as e:
            logger.warning("Failed to remove vectorstore directory %s: %s", persist_directory, e)

    os.makedirs(persist_directory, exist_ok=True)
    tmp_dir = os.path.join("data", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_docs: List[Document] = []

    for uploaded_file in uploaded_files:
        # save temporary file
        filename = getattr(uploaded_file, "name", "uploaded.pdf")
        temp_path = os.path.join(tmp_dir, filename)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        docs = _load_pdf_safe(temp_path, splitter)
        if not docs:
            logger.warning("No documents extracted from %s", filename)
        else:
            # add source metadata
            for d in docs:
                if d.metadata is None:
                    d.metadata = {}
                d.metadata["source"] = filename
            all_docs.extend(docs)

    if not all_docs:
        raise ValueError("No valid documents could be extracted from uploaded files.")

    # embeddings (Hugging Face, light and good for QA/search)
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    except Exception as e:
        logger.exception("Failed to initialize HuggingFaceEmbeddings: %s", e)
        raise

    # create chroma vectorstore from documents
    try:
        vectordb = Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        # Chroma persistence is automatic in newer versions, but call persist if method exists
        try:
            vectordb.persist()
        except Exception:
            pass

    except Exception as e:
        logger.exception("Failed to create Chroma vectorstore: %s", e)
        raise

    logger.info("Built vectorstore '%s' at %s with %d documents", collection_name, persist_directory, len(all_docs))
    return vectordb
