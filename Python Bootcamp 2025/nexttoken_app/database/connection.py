import sqlite3

import config
from .schema import SCHEMA


def get_connection() -> sqlite3.Connection:
    # config.DB_PATH is read at call time so tests can point it elsewhere
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)
