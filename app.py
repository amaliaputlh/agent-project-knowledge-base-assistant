# app.py
import streamlit as st
import os

# === LangChain & LangGraph Imports ===
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

# === Local Modules ===
from modules.vectorstore_manager import persist_documents_to_chroma
from modules.rag_tools import DocumentRetrieverTool

# ===========================
# 1️⃣ Streamlit Page Setup
# ===========================
st.set_page_config(page_title="📚 Company Knowledge Assistant", layout="wide")
st.title("💬 Company Knowledge Base Assistant")
st.caption("Ask questions based on internal company documents (PDF).")

# ===========================
# 2️⃣ Sidebar Settings
# ===========================
with st.sidebar:
    st.subheader("⚙️ Settings")

    google_api_key = st.text_input("🔑 Google AI API Key (optional for chat)", type="password")
    reset_button = st.button("♻️ Reset Conversation", help="Clear memory and restart the chat")

    st.divider()
    st.subheader("📂 Upload Company Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
    )

# ===========================
# 3️⃣ Load PDFs & Build Vectorstore
# ===========================
# --- after uploading and building vectorstore ---
if uploaded_files:
    with st.spinner("Processing uploaded PDFs and updating knowledge base..."):
        vectordb = persist_documents_to_chroma(uploaded_files, persist_directory=None)  # None => in-memory safe for Streamlit Cloud
        # Save the vectordb into session state so the retriever tool can access it
        st.session_state['vectordb'] = vectordb
        st.success(f"✅ Knowledge base updated with {len(uploaded_files)} file(s).")

# ===========================
# 4️⃣ Initialize Agent (LangGraph)
# ===========================

if not google_api_key:
    st.warning(
        "⚠️ Please enter your **Google API Key** to start chatting.\n\n"
        "This key is required to activate the AI assistant for reasoning. "
        "You can obtain it from [Google AI Studio](https://aistudio.google.com/app/apikey).",
        icon="⚠️"
    )
    st.stop()
else:
    st.info("💬 You can now start asking questions about your company documents!")

if (
    "agent" not in st.session_state
) or (getattr(st.session_state, "_last_key", None) != google_api_key):
    try:
        # --- LLM Initialization (still using Gemini for reasoning)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
            temperature=0.3,
        )

        # --- Tool Setup (local embedding retriever)
        retriever_tool = DocumentRetrieverTool()

        # --- Agent Creation ---
        st.session_state.agent = create_react_agent(
            model=llm,
            tools=[retriever_tool],
            prompt=(
                "You are an internal company assistant. "
                "If the user's question is related to company documents, "
                "you MUST use the `document_retriever` tool to get relevant information "
                "before answering. Do not guess without retrieving context first. "
            ),
        )

        st.session_state._last_key = google_api_key
        st.session_state.pop("messages", None)

    except Exception as e:
        st.error(
            f"❌ Unable to initialize the assistant. Please check your API key or connection.\n\n**Details:** {e}"
        )
        st.stop()


# ===========================
# 5️⃣ Reset Conversation
# ===========================
if reset_button:
    st.session_state.pop("agent", None)
    st.session_state.pop("messages", None)
    st.success("Conversation reset successfully.")
    st.rerun()

# ===========================
# 6️⃣ Chat History Display
# ===========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ===========================
# 7️⃣ Chat Input Handling
# ===========================
prompt = st.chat_input("Ask a question about company documents...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        response = st.session_state.agent.invoke({"messages": messages})

        if "messages" in response and len(response["messages"]) > 0:
            answer = response["messages"][-1].content
        else:
            answer = "I'm sorry, I couldn't generate a response."

    except Exception as e:
        answer = f"❌ Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
