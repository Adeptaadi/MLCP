import streamlit as st
from mode1 import run_mode1
from mode2 import run_mode2
st.set_page_config(page_title="Explainable AI System", layout="wide")

st.title("🔍 Explainable AI Dashboard")

mode = st.sidebar.radio(
    "Select Mode",
    ["Mode 1: Prediction", "Mode 2: Dataset Analysis"]
)

if mode == "Mode 1: Prediction":
    run_mode1()
else:
    run_mode2()