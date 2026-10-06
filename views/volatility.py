import streamlit as st

from ui import data, layout

st.title("Implied volatility")
st.caption("What NSE option prices say about expected volatility, from each trading day's official closes.")
c1, _ = st.columns([1, 3])
c1.selectbox("Underlying", data.UNDERLYINGS, key="vol_underlying")
layout.tabs({"Surface": "surface.py", "History": "history.py"})
