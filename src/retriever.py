from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st


# Initialize embeddings & documents
@st.cache_resource
def load_retriever():
    """
    This app's retriever logic
    input: None
    output: retriever obj
    """

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("./vectorstore", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    return retriever
