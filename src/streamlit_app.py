import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import InferenceClient
import re


HF_TOKEN = os.environ.get("HF_TOKEN")

# ----------------------


# qa_template = """Use the given context to answer the question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Keep the answer as concise as possible.

# Context: {context}

# Question: {question}
# Answer:
# """


# prompt = PromptTemplate.from_template(qa_template)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a knowledgeable agricultural research assistant.\n"
        "Use the context below to answer the question concisely.\n"
        "Respond ONLY with the final answer inside <answer> and </answer> tags.\n\n"
        "Example:\n"
        "Question: What is photosynthesis?\n"
        "Answer: <answer>Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll, water, and carbon dioxide.</answer>\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    )


# Initialize embeddings & documents
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("./vectorstore", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    return retriever

# Load a lightweight model via HuggingFace pipeline
@st.cache_resource
def load_llm():
    # pipe = pipeline("text-generation", model="google/flan-t5-small", max_new_tokens=256)
    # load the tokenizer and model on cpu/gpu

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", load_in_8bit=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    
    return HuggingFacePipeline(pipeline=pipe)

# Setup RAG Chain
@st.cache_resource
def setup_qa():

    retriever = load_retriever()
    llm = load_llm().bind(stop=["</answer>"])
    question_answer_chain = create_stuff_documents_chain(llm,prompt)
    # chain = create_retrieval_chain(retriever, question_answer_chain)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True,chain_type_kwargs={'prompt':prompt})
    return qa_chain


# Streamlit App UI
st.title("🌾 AgriQuery: RAG-Based Research Assistant")

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
    # st.success(answer[-1])
    # st.success(answer)
    st.success(f"Source Document(s): {result['source_documets']}")

