import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from modules.vectorstore_manager import persist_documents_to_chroma
from modules.rag_tools import DocumentRetrieverTool

st.set_page_config(page_title="📚 Company Knowledge Assistant", layout="wide")
st.title("💬 Company Knowledge Base Assistant")
st.caption("Ask questions based on internal company documents (PDF).")

# ======================
# Sidebar Configuration
# ======================
with st.sidebar:
    st.subheader("⚙️ Settings")
    google_api_key = st.text_input("🔑 Google API Key", type="password")
    reset_button = st.button("♻️ Reset Conversation")

    st.divider()
    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Set env for embeddings
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

# ======================
# PDF → Vectorstore
# ======================
if uploaded_files:
    with st.spinner("Processing documents..."):
        vectordb = persist_documents_to_chroma(uploaded_files)
        st.success(f"✅ Knowledge base updated with {len(uploaded_files)} file(s).")

# ======================
# API Key Check
# ======================
if not google_api_key:
    st.warning(
        "⚠️ Please enter your **Google API Key** to start chatting.\n\n"
        "Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).",
        icon="⚠️"
    )
    st.stop()
else:
    st.info("💬 You can now start asking questions!")

# ======================
# Initialize Agent
# ======================
if "agent" not in st.session_state or st.session_state.get("_last_key") != google_api_key:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
            temperature=0.3,
        )
        retriever_tool = DocumentRetrieverTool()

        st.session_state.agent = create_react_agent(
            model=llm,
            tools=[retriever_tool],
            prompt=(
                "You are an internal company assistant. "
                "If the user's question relates to company documents, "
                "always use the `document_retriever` tool before answering. "
                "Never guess without context."
            ),
        )

        st.session_state._last_key = google_api_key
        st.session_state.pop("messages", None)

    except Exception as e:
        st.error(f"❌ Could not initialize assistant: {e}")
        st.stop()

# ======================
# Chat Interface
# ======================
if reset_button:
    st.session_state.clear()
    st.success("🔄 Chat reset successfully.")
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about company documents...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        messages = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in st.session_state.messages
        ]

        response = st.session_state.agent.invoke({"messages": messages})
        answer = response["messages"][-1].content if "messages" in response else "No response generated."

    except Exception as e:
        answer = f"❌ Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
