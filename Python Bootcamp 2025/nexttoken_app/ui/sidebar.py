import streamlit as st

import config
from database import repository
from services.document_loader import extract_text, file_hash
from services.text_processing import tokenize


def render_sidebar() -> None:
    with st.sidebar:
        st.header("1. Upload document")
        upload = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "doc", "txt"])
        if upload is not None and st.button("Create / open session", type="primary"):
            try:
                data = upload.getvalue()
                text = extract_text(upload.name, data)
                if len(tokenize(text)) < config.MIN_TOKENS:
                    st.error("Could not extract enough text (scanned PDF?).")
                else:
                    sid, created = repository.get_or_create_session(upload.name, file_hash(data), text)
                    st.session_state.sid = sid
                    st.success("New session created." if created
                               else "This document already has a session; reopened it.")
            except Exception as e:
                st.error(str(e))

        st.divider()
        st.header("Sessions")
        sessions = repository.list_sessions()
        if not sessions:
            st.info("No sessions yet.")
        for s in sessions:
            label = f"#{s['id']} · {s['filename']}" + (f" ({s['model_type']})" if s["model_type"] else "")
            if st.button(label, key=f"open_{s['id']}", use_container_width=True):
                st.session_state.sid = s["id"]
