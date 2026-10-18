import streamlit as st

import config
from database import repository
from services.trainer import train_session


def render_train_tab(session):
    c1, c2, c3 = st.columns(3)
    model_type = c1.selectbox("Model", config.MODEL_TYPES)
    epochs = c2.number_input("Epochs", 1, 300, 30, step=5)
    ctx_len = c3.number_input("Context length (tokens)", 3, 60, 15)

    if st.button("Train model"):
        bar, status = st.progress(0.0), st.empty()

        def on_epoch(epoch, total, logs):
            bar.progress(epoch / total)
            status.write(f"Epoch {epoch}/{total} · loss={logs.get('loss', 0):.3f} · "
                         f"acc={logs.get('accuracy', 0):.3f}")

        try:
            with st.spinner("Training…"):
                n = train_session(session["id"], model_type, int(epochs), int(ctx_len), on_epoch)
            st.success(f"Trained on {n:,} samples.")
            session = repository.get_session(session["id"])
        except Exception as e:
            st.error(str(e))

    if session["model_path"]:
        st.write(f"**Current model:** {session['model_type']} · epochs={session['epochs']} · "
                 f"loss={session['final_loss']:.3f} · accuracy={session['final_acc']:.3f}")
