## AGRIQUERY - RAG-LLM Powered Q&A App for Agricultural Researchers.

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
│   ├── rag_chain.py         # LangChain RAG setup
│   ├── ingest_docs.py       # PDF/text ingestion and embedding
│   ├── llm_config.py        # LLM + embedding setup
│   └── main_app.py          # Streamlit or Gradio UI
│
├── data/
│   ├── raw/                 # Original documents (PDFs, TXT)
│   └── processed/           # Preprocessed chunks (JSONs, Pickle, etc.)
│
├── logs/                    # Monitoring logs (latency, tokens)
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml   # (Optional, if adding vector store separately)
│
├── notebooks/
│   └── experiments.ipynb    # Testing LangChain chains and embeddings
│
├── tests/
│   └── test_chain.py        # Simple unit tests for chain and retrieval
│
├── requirements.txt
├── .env                     # API keys
├── README.md
└── architecture.png         # Architecture diagram for the README


```
pip install transformers langchain langchain_community sentence-transformers faiss-cpu

## 🏁 Complete Local RAG Stack
Component	       Tool
Chunking	LangChain Splitters
Embedding	Sentence Transformers
Vector DB	FAISS
LLM	        HuggingFace Llama (Locally)
RAG Chain	LangChain RetrievalQA
