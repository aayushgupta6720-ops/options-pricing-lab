import streamlit as st

from ui import layout

st.title("Option pricer")
st.caption("One option, set in the sidebar, priced and taken apart several ways.")
layout.tabs(
    {
        "Prices": "prices.py",
        "Greeks": "greeks.py",
        "Convergence": "convergence.py",
        "Heston model": "heston.py",
    }
)
