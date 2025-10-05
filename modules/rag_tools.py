# modules/rag_tools.py
import logging
from typing import Optional

from langchain.tools import BaseTool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# We'll import streamlit inside the method to avoid import-time dependency when used in other contexts
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DocumentRetrieverTool(BaseTool):
    """
    Tool wrapper that searches the Chroma vectorstore and returns short snippets + source.
    It first looks for an in-memory vectordb placed in streamlit session state under 'vectordb'.
    If not found and a persist_directory is configured, it will load Chroma from disk.
    """

    name: str = "document_retriever"
    description: str = "Search uploaded company documents and return relevant snippets with source metadata."
    embedding_model: str = "multi-qa-MiniLM-L6-cos-v1"
    top_k: int = 5
    persist_directory: Optional[str] = None  # optional, used only if vectordb not in session_state

    def _get_vectorstore(self):
        """
        Try to obtain the vectorstore in this order:
        1. streamlit.session_state['vectordb'] (recommended; in-memory)
        2. load Chroma from persist_directory if specified
        3. return None
        """
        try:
            import streamlit as st
        except Exception:
            st = None

        # 1) check session_state vectordb
        if st is not None:
            vectordb = st.session_state.get("vectordb")
            if vectordb:
                logger.info("Using vectordb from Streamlit session_state.")
                return vectordb

        # 2) try to load from disk if persist_directory provided
        if self.persist_directory:
            try:
                embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
                db = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings)
                logger.info("Loaded Chroma vectorstore from persist_directory: %s", self.persist_directory)
                return db
            except Exception as e:
                logger.warning("Failed to load Chroma from persist_directory: %s", e)
                return None

        return None

    def _run(self, query: str) -> str:
        try:
            db = self._get_vectorstore()
            if db is None:
                return "I cannot search documents right now because no knowledge base is loaded. Please upload PDF files first."

            # prefer retriever interface if available
            try:
                retriever = db.as_retriever(search_kwargs={"k": self.top_k})
                docs = retriever.get_relevant_documents(query)
            except Exception:
                docs = db.similarity_search(query, k=self.top_k)

            logger.info("[RAG DEBUG] Retrieved %d docs for query: %s", len(docs), query)

            if not docs:
                return "I am sorry — I could not find relevant information in the uploaded documents."

            parts = []
            for d in docs:
                src = d.metadata.get("source", "unknown")
                snippet = d.page_content.replace("\n", " ").strip()
                if len(snippet) > 800:
                    snippet = snippet[:800].rsplit(" ", 1)[0] + "..."
                parts.append(f"Source: {src}\n{snippet}")

            return "\n\n---\n\n".join(parts)

        except Exception as e:
            logger.exception("Error in DocumentRetrieverTool: %s", e)
            return f"❌ Error while retrieving documents: {e}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
