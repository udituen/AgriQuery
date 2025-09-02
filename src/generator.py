from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.llms import Ollama
import streamlit as st

# Load a lightweight model via HuggingFace pipeline
@st.cache_resource
def load_llm():
    """
    This file the generator logic using meta's Llama LLM
    """


    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", load_in_8bit=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    
    return HuggingFacePipeline(pipeline=pipe)
