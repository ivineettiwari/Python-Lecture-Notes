"""All SQL lives here."""
import json
import os
from datetime import datetime
from typing import Optional

from .connection import get_connection


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- sessions
def get_or_create_session(filename: str, file_hash: str, text: str):
    """Return (session_id, created). One session per unique document."""
    with get_connection() as c:
        row = c.execute("SELECT id FROM sessions WHERE file_hash=?", (file_hash,)).fetchone()
        if row:
            return row["id"], False
        cur = c.execute(
            "INSERT INTO sessions (filename, file_hash, text, created_at) VALUES (?,?,?,?)",
            (filename, file_hash, text, _now()),
        )
        return cur.lastrowid, True


def get_session(session_id: int):
    with get_connection() as c:
        return c.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()


def list_sessions():
    with get_connection() as c:
        return c.execute(
            "SELECT id, filename, model_type, created_at FROM sessions ORDER BY id DESC"
        ).fetchall()


def save_training(session_id, vocab: dict, ctx_len, model_type, epochs, loss, acc, model_path):
    with get_connection() as c:
        c.execute(
            """UPDATE sessions SET vocab_json=?, ctx_len=?, model_type=?, epochs=?,
               final_loss=?, final_acc=?, model_path=? WHERE id=?""",
            (json.dumps(vocab), ctx_len, model_type, epochs, loss, acc, str(model_path), session_id),
        )


def delete_session(session_id: int) -> None:
    session = get_session(session_id)
    if session and session["model_path"] and os.path.exists(session["model_path"]):
        os.remove(session["model_path"])
    with get_connection() as c:
        c.execute("DELETE FROM generations WHERE session_id=?", (session_id,))
        c.execute("DELETE FROM sessions WHERE id=?", (session_id,))


# ------------------------------------------------------------- generations
def save_generation(session_id, seed, num_tokens, temperature, output) -> None:
    with get_connection() as c:
        c.execute(
            "INSERT INTO generations (session_id, seed, num_tokens, temperature, output, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (session_id, seed, num_tokens, temperature, output, _now()),
        )


def list_generations(session_id: int):
    with get_connection() as c:
        return c.execute(
            "SELECT * FROM generations WHERE session_id=? ORDER BY id DESC", (session_id,)
        ).fetchall()
