import os
import logging
from langchain.tools import BaseTool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DocumentRetrieverTool(BaseTool):
    """A LangChain tool that retrieves relevant snippets from Chroma vectorstore."""

    name: str = "document_retriever"
    description: str = "Search uploaded company documents and return relevant snippets."
    persist_directory: str = "data/vectorstore"
    collection_name: str = "kb_collection"
    top_k: int = 5

    def _run(self, query: str) -> str:
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                return "⚠️ Missing Google API key. Please configure it first."

            # ✅ Use Gemini Embedding model (fast & Streamlit-compatible)
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-exp-03-07",
                google_api_key=api_key,
            )

            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings,
                collection_name=self.collection_name,
            )

            retriever = db.as_retriever(search_kwargs={"k": self.top_k})
            docs = retriever.get_relevant_documents(query)

            if not docs:
                return "I couldn’t find relevant information in the uploaded documents."

            results = []
            for d in docs:
                src = d.metadata.get("source", "unknown")
                snippet = d.page_content.replace("\n", " ").strip()
                snippet = (snippet[:800].rsplit(" ", 1)[0] + "...") if len(snippet) > 800 else snippet
                results.append(f"📄 **Source:** {src}\n{snippet}")

            logger.info("[RAG] Retrieved %d docs for query: %s", len(docs), query)
            return "\n\n---\n\n".join(results)

        except Exception as e:
            logger.exception("Error in DocumentRetrieverTool: %s", e)
            return f"❌ Retrieval error: {e}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
