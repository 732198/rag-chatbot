import os

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()
DATA_DIR = "data"
Settings.llm = GoogleGenAI(model="gemini-3.8-flash")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


@st.cache_resource
def get_query_engine():
    if not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY not found. Add it to your .env file.")
        st.stop()
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    return index.as_query_engine()


st.title("Babson Handbook Chatbot")
query_engine = get_query_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input("Ask about the handbook")
if prompt:
    st.chat_message("user").write(prompt)
    response = query_engine.query(prompt)
    bot_response = response.response
    st.chat_message("assistant").write(bot_response)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": bot_response})