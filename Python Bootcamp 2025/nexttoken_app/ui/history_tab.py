import streamlit as st

from database import repository


def _delete(sid):
    repository.delete_session(sid)
    st.session_state.sid = None


def render_history_tab(session):
    gens = repository.list_generations(session["id"])
    if not gens:
        st.info("No generations yet.")
    for g in gens:
        st.markdown(f"**{g['created_at']}** · seed: `{g['seed']}` · "
                    f"tokens: {g['num_tokens']} · temp: {g['temperature']}")
        st.write(g["output"])
        st.divider()
    st.button("🗑️ Delete this session", on_click=_delete, args=(session["id"],))
