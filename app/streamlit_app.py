"""
This file contains the user interface app
"""
import streamlit as st
import os
import torch
from huggingface_hub import InferenceClient
import re
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0,parent_dir)

from src.rag_pipeline import setup_qa

EXAMPLE_QUESTIONS = [
    "What is agriculture?",
    "Why is crop rotation important?",
    "How does composting help farming?",
]

HF_TOKEN = os.environ.get("HF_TOKEN")

st.title("🌾 AgriQuery: RAG-Based Research Assistant")

# Show example questions
with st.expander("💡 Try example questions"):
    for q in EXAMPLE_QUESTIONS:
        st.markdown(f"- {q}")

query = st.text_input("Ask a question related to agriculture:")

if query:
    qa = setup_qa()
    with st.spinner("Thinking..."):
        result = qa.invoke({"query":query})
        raw = result["result"]
        raw_answer = result["result"]

    matches = re.findall(r"<answer>(.*?)</answer>", raw_answer, re.DOTALL)

    if matches:
        clean_answer = matches[-1].strip()   # last <answer>...</answer> block
    else:
        clean_answer = raw_answer.strip()    # fallback

    st.success(clean_answer)
    st.success(f"Source Document(s): {result['source_documents']}")

