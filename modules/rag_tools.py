# modules/rag_tools.py
import logging
from typing import Optional

from langchain.tools import BaseTool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DocumentRetrieverTool(BaseTool):
    """
    A LangChain tool wrapper that searches the Chroma vectorstore and returns snippets.
    """

    name: str = "document_retriever"  # valid name for Gemini/function-tool usage (no spaces)
    description: str = "Search company documents and return relevant snippets with source metadata."
    persist_directory: str = "data/vectorstore"
    collection_name: str = "kb_collection"
    embedding_model: str = "multi-qa-MiniLM-L6-cos-v1"
    top_k: int = 5

    def _run(self, query: str) -> str:
        try:
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            # load chroma with same embedding function to ensure compatibility
            db = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings, collection_name=self.collection_name)

            # prefer retriever interface if available
            try:
                retriever = db.as_retriever(search_kwargs={"k": self.top_k})
                docs = retriever.get_relevant_documents(query)
            except Exception:
                # fallback: similarity_search on vectorstore
                docs = db.similarity_search(query, k=self.top_k)

            logger.info("[RAG DEBUG] Retrieved %d docs for query: %s", len(docs), query)

            if not docs:
                return "I am sorry, but I could not find any relevant information in the uploaded company documents."

            results = []
            for d in docs:
                src = d.metadata.get("source", "unknown")
                snippet = d.page_content.replace("\n", " ").strip()
                # limit snippet length
                if len(snippet) > 800:
                    snippet = snippet[:800].rsplit(" ", 1)[0] + "..."
                results.append(f"Source: {src}\n{snippet}")

            return "\n\n---\n\n".join(results)

        except Exception as e:
            logger.exception("Error in DocumentRetrieverTool: %s", e)
            return f"❌ Error while retrieving documents: {e}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
