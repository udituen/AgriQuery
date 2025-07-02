
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI()

# Dummy search endpoint
@app.get("/query")
def query_agri_knowledge(q: str = Query(..., description="Your research question")):
    # Replace with real RAG pipeline later
    return {"question": q, "answer": "This is a placeholder answer from AgriQuery."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
