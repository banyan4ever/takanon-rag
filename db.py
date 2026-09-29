import psycopg
from psycopg.rows import dict_row

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sections (
    number     TEXT PRIMARY KEY,
    chapter    TEXT NOT NULL,
    text       TEXT NOT NULL,
    version    INTEGER NOT NULL DEFAULT 1,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def connect():
    # DATABASE_URL is passed through untouched, including ?sslmode=require
    if not config.DATABASE_URL:
        raise RuntimeError("DATABASE_URL is missing from .env")
    conn = psycopg.connect(config.DATABASE_URL, row_factory=dict_row, autocommit=True, connect_timeout=10)
    conn.execute(SCHEMA)
    return conn


def upsert_section(conn, number, chapter, text):
    """Insert a new section (version 1) or update it and bump its version."""
    return conn.execute(
        """
        INSERT INTO sections (number, chapter, text) VALUES (%s, %s, %s)
        ON CONFLICT (number) DO UPDATE
            SET chapter = EXCLUDED.chapter,
                text = EXCLUDED.text,
                version = sections.version + 1,
                updated_at = now()
        RETURNING *
        """,
        (number, chapter, text),
    ).fetchone()


def get_sections(conn, numbers):
    return conn.execute(
        "SELECT * FROM sections WHERE number = ANY(%s) ORDER BY number", (list(numbers),)
    ).fetchall()


def count_sections(conn):
    return conn.execute("SELECT count(*) AS n FROM sections").fetchone()["n"]


def delete_section(conn, number):
    return conn.execute("DELETE FROM sections WHERE number = %s", (number,)).rowcount
