import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

"""
# Welcome to AgriQuery!

"""

st.title("AgriQuery - Agricultural AI Assistant")

query = st.text_input("Ask a question about crops, soil, weather, etc.")

