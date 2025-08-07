# source:https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from fastapi import FastAPI
import requests
from pydantic import BaseModel
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os


load_dotenv()
token= os.getenv("TOKEN")
app = FastAPI()

class QueryInput(BaseModel):
    query: str


# build the retrieval and augmented generator chain here

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("./vectorstore/agriquery_faiss_index", embeddings, allow_dangerous_deserialization=True)
llm = Ollama(model="llama3", base_url="http://localhost:11434")

retriever = db.as_retriever()

system_prompt = (
    "You are an agriultural research assistant."
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm,prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)


@app.post("/query")
async def query_handler(input: QueryInput):
    result = chain.invoke({"input": input.query})
    answer = result['answer'].replace("\\n", "\n").strip()

    return {"answer": answer}