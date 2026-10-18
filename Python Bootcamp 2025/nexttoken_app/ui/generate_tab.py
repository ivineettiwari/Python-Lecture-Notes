import os

import streamlit as st

from database import repository
from services.generator import generate_text
from services.text_processing import tokenize


def render_generate_tab(session):
    if not session["model_path"] or not os.path.exists(session["model_path"]):
        st.warning("Train a model for this session first.")
        return

    seed = st.text_input("Seed text", "machine learning")
    c1, c2 = st.columns(2)
    n_tokens = c1.number_input("Number of tokens to generate", 1, 500, 10, step=1)
    temperature = c2.slider("Temperature (0 = greedy)", 0.0, 1.5, 0.7, 0.1)

    if st.button("Generate", type="primary"):
        if not tokenize(seed):
            st.error("Enter a seed text.")
            return
        with st.spinner("Generating…"):
            result = generate_text(session, seed, int(n_tokens), float(temperature))
        repository.save_generation(session["id"], seed, int(n_tokens), float(temperature), result)
        st.success(result)
