SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    file_hash   TEXT UNIQUE NOT NULL,
    text        TEXT NOT NULL,
    vocab_json  TEXT,
    ctx_len     INTEGER,
    model_type  TEXT,
    epochs      INTEGER,
    final_loss  REAL,
    final_acc   REAL,
    model_path  TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seed        TEXT NOT NULL,
    num_tokens  INTEGER NOT NULL,
    temperature REAL NOT NULL,
    output      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_generations_session ON generations(session_id);
"""
