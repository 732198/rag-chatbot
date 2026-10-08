import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()

DATA_DIR = Path("data")


def get_api_key() -> str:
    """Validate that the GEMINI_API_KEY environment variable is set."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "MISSING API KEY: GEMINI_API_KEY was not found in your environment. "
            "Please add GEMINI_API_KEY=your_key_here to your .env file."
        )
        st.stop()
    return api_key


def validate_data_dir(data_dir: Path):
    """Validate that the data directory exists, is a folder, and contains files."""
    if not data_dir.exists() or not data_dir.is_dir():
        st.error(
            f"DATA DIRECTORY NOT FOUND: Expected directory '{data_dir.resolve()}' does not exist or is not a folder. "
            f"Please create a folder named '{data_dir.name}' in the root directory."
        )
        st.stop()

    # Filter out hidden files like .DS_Store
    valid_files = [f for f in data_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
    if not valid_files:
        st.error(
            f"DATA DIRECTORY IS EMPTY: No valid documents were found inside '{data_dir.resolve()}'. "
            f"Please add at least one document to the '{data_dir.name}' folder."
        )
        st.stop()


@st.cache_resource
def get_query_engine(api_key: str):
    """Load settings, parse documents, build the index, and return the query engine."""
    # Pass validated API key directly to GoogleGenAI
    Settings.llm = GoogleGenAI(model="gemini-2.5-flash", api_key=api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Load documents using DATA_DIR constant
    documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()
    
    # VectorStoreIndex.from_documents splits into chunks before embedding
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    return index.as_query_engine()


# --- Streamlit Layout & Logic ---

st.title("Babson Handbook Chatbot")

# 1. Fail-fast validation checks before calling cached function
api_key = get_api_key()
validate_data_dir(DATA_DIR)

# 2. Safely initialize query engine with try-except
try:
    query_engine = get_query_engine(api_key)
except Exception as e:
    st.error(f"ENGINE INITIALIZATION FAILED: Unable to build index from documents. Details: {e}")
    st.stop()

# 3. Chat Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# Handle user query input
prompt = st.chat_input("Ask about the handbook")
if prompt:
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        with st.spinner("Searching handbook..."):
            response = query_engine.query(prompt)
            bot_response = response.response

        st.chat_message("assistant").write(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
    except Exception as e:
        st.error(
            f"QUERY FAILED: Could not complete request due to a network or model error: {e}. "
            "Please check your connection and try asking again."
        )