---
title: Agriquery
emoji: 🦀
colorFrom: green
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
- RAG
- Ollama
- FAISS

pinned: false
short_description: LLM-powered question-answering system using RAG
---

### AGRIQUERY - RAG-LLM Powered Q&A App for Agricultural Researchers.

AgriQuery is an LLM-powered Q&A system built for agricultural researchers. It processes scientific publications and enables users to ask natural language questions, receiving context-aware answers backed by retrieved text. Built with LangChain, FAISS, Airflow, and Docker, it demonstrates a production-ready RAG architecture for domain-specific information retrieval.

### End-to-End RAG with Airflow, FAISS, Llama, FastAPI

## Usage
1. `docker-compose up --build`
2. Access:
   - Airflow: http://localhost:8080
   - FastAPI: http://localhost:8000/query?question=Your+question+here
3. Run DAG `rag_pipeline` in Airflow UI to ingest & build FAISS.
4. Query the running FastAPI endpoint.


```
agriquery-rag-llm/
│
├── app/
│   ├── rag.py               # LangChain RAG setup
│   └── main_app.py          # Streamlit or Gradio UI
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml   
│
├── src/
│   └── experiments.ipynb    # Testing LangChain chains and embeddings
│   └── prepocess.py         # pdf upload, chunking and embedding setup
├── requirements.txt
├── .env                     # API keys
├── README.md


```
pip install transformers langchain langchain_community sentence-transformers faiss-cpu

## Complete Local RAG Stack
Component	       Tool
Chunking	LangChain Splitters
Embedding	Sentence Transformers
Vector DB	FAISS
LLM	        HuggingFace Llama (Locally)
RAG Chain	LangChain RetrievalQA
