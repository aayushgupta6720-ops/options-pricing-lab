import streamlit as st

from ui import layout

st.title("Models vs market")
st.caption(
    "Pricing models fitted to every trading day's NSE closes, and where each one matches the market or misses."
)
layout.tabs({"Heston & SABR": "heston_sabr.py", "Rough volatility": "rough.py"})
