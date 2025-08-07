import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Initialize embeddings & documents
# ----------------------
@st.cache_resource
def load_retriever():
    # Load documents
    with open("data/docs.txt", "r") as f:
        docs = f.read().split("\n")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(docs, embeddings)
    retriever = db.as_retriever()
    return retriever

# Load a lightweight model via HuggingFace pipeline
# ----------------------
@st.cache_resource
def load_llm():
    pipe = pipeline("text-generation", model="google/flan-t5-small", max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

# Setup RAG Chain
# ----------------------
@st.cache_resource
def setup_qa():
    retriever = load_retriever()
    llm = load_llm()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain


# Streamlit App UI
# ----------------------
st.title("AgriQuery: RAG Demo (Streamlit + HF)")

query = st.text_input("Ask a question:")

if query:
    qa = setup_qa()
    with st.spinner("Thinking..."):
        result = qa.run(query)
    st.success(result)
