"""
script to test the langchain retriever, and Llama generation model

"""

import pytest
from rag_app import answer_question


# test for null entries
def test_answer_non_empty():
    query = "What is agriculture?"
    response = answer_question(query)
    assert isinstance(response, str)
    assert len(response) > 0


# check for gibberish or none english in answer
def test_answer_relevance():
    query = "what is agriculture?"
    response = answer_question(query)
    assert "Guido" in response or "Rossum" in response

@pytest.mark.parametrize("query", [
    "what is soil erosion?",
    "what is climae change?",
    "What is agriculture?"
])

# check for multiple entries
def test_multiple_queries(query):
    response = answer_question(query)
    assert isinstance(response, str)
    assert len(response) > 0
