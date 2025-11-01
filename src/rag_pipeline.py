from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import RetrievalQA
import streamlit as st

from src.generator import load_llm
from src.retriever import load_retriever


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


# Setup RAG Chain
@st.cache_resource
def setup_qa():
    """
    This function contains the rag pipeline
    input: generator, retriever
    output: pipeline
    """


    retriever = load_retriever()
    llm = load_llm().bind(stop=["</answer>"])
    question_answer_chain = create_stuff_documents_chain(llm,prompt)
    # chain = create_retrieval_chain(retriever, question_answer_chain)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True,chain_type_kwargs={'prompt':prompt})
    return qa_chain

