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


def get_query_engine():
    documents = SimpleDirectoryReader("data/").load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()

st.title("Babson Handbook Chatbot")
query_engine = get_query_engine()   
prompt = st.chat_input("Ask about the handbook")
if prompt:
    st.write(f"User:{prompt}")
    response = query_engine.query(prompt)
    bot_response = response.response
    st.write(f"Bot: {bot_response}")