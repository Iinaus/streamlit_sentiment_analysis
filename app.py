import streamlit as st
import pandas as pd

st.write("""
# My first app
Is the pipeline working?
""")

df = pd.read_csv("data.csv", delimiter=";")
st.line_chart(df['sum'])