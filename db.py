import psycopg
from psycopg.rows import dict_row

import config

# One row per chunk of a regulation section; (number, chunk_index) is unique.
SCHEMA = """
CREATE TABLE IF NOT EXISTS sections (
    number      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    chapter     TEXT NOT NULL,
    text        TEXT NOT NULL,
    page        INTEGER,
    summary     TEXT,
    version     INTEGER NOT NULL DEFAULT 1,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (number, chunk_index)
)
"""
# version that was last sent to Pinecone; a row needs (re-)embedding while this differs from `version`
MIGRATION = "ALTER TABLE sections ADD COLUMN IF NOT EXISTS embedded_version INTEGER"


def connect():
    # DATABASE_URL is passed through untouched, including ?sslmode=require
    if not config.DATABASE_URL:
        raise RuntimeError("DATABASE_URL is missing from .env")
    conn = psycopg.connect(config.DATABASE_URL, row_factory=dict_row, autocommit=True, connect_timeout=10)
    conn.execute(SCHEMA)
    conn.execute(MIGRATION)
    return conn


def upsert_section(conn, number, chapter, text, chunk_index=0, page=None, summary=None):
    """Insert a chunk (version 1) or update it and bump its version.

    Returns {number, chunk_index, version, inserted}, or None if the row already held exactly this content.
    """
    return conn.execute(
        """
        INSERT INTO sections (number, chunk_index, chapter, text, page, summary)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (number, chunk_index) DO UPDATE
            SET chapter = EXCLUDED.chapter,
                text = EXCLUDED.text,
                page = EXCLUDED.page,
                summary = EXCLUDED.summary,
                version = sections.version + 1,
                updated_at = now()
            WHERE (sections.chapter, sections.text, sections.page, sections.summary)
                IS DISTINCT FROM (EXCLUDED.chapter, EXCLUDED.text, EXCLUDED.page, EXCLUDED.summary)
        RETURNING number, chunk_index, version, (xmax = 0) AS inserted
        """,
        (number, chunk_index, chapter, text, page, summary),
    ).fetchone()


def get_sections(conn, numbers):
    return conn.execute(
        "SELECT * FROM sections WHERE number = ANY(%s) ORDER BY number, chunk_index", (list(numbers),)
    ).fetchall()


def count_sections(conn):
    return conn.execute("SELECT count(*) AS n FROM sections").fetchone()["n"]


def delete_section(conn, number):
    return conn.execute("DELETE FROM sections WHERE number = %s", (number,)).rowcount
