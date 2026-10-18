"""Streamlit entry point:  streamlit run app.py"""
import streamlit as st

from database import init_db, repository
from services.text_processing import tokenize
from ui.generate_tab import render_generate_tab
from ui.history_tab import render_history_tab
from ui.sidebar import render_sidebar
from ui.train_tab import render_train_tab

st.set_page_config(page_title="Next Token Generator", page_icon="📝", layout="wide")
init_db()

st.title("📝 Next-Token Generator")
st.caption("Upload a document → train an RNN/GRU/LSTM on it → generate the next N tokens.")

st.session_state.setdefault("sid", None)
render_sidebar()

sid = st.session_state.sid
session = repository.get_session(sid) if sid else None
if session is None:
    st.info("Upload a document in the sidebar, or pick an existing session.")
    st.stop()

st.subheader(f"Session #{session['id']} — {session['filename']}")
st.caption(f"Created {session['created_at']} · {len(tokenize(session['text'])):,} tokens")
with st.expander("Document preview"):
    st.text(session["text"][:3000] + ("…" if len(session["text"]) > 3000 else ""))

tab_train, tab_gen, tab_hist = st.tabs(["2. Train", "3. Generate", "History"])
with tab_train:
    render_train_tab(session)
with tab_gen:
    render_generate_tab(session)
with tab_hist:
    render_history_tab(session)
