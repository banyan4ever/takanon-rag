"""Embed Neon sections that aren't in Pinecone yet (or changed since) and upsert them into the existing index.

Run: .venv/bin/python add_to_pinecone.py   (safe to re-run or resume after a crash)
"""
import logging
import sys
import time
from pathlib import Path

import config
import db
import vectors

BATCH_SIZE = 100
LOG_PATH = Path(__file__).parent / "add_to_pinecone.log"

log = logging.getLogger("add_to_pinecone")


def retry(what, fn, attempts=6):
    """Call fn(), retrying API errors with backoff (5, 10, 20, 40, 60s) — enough to outlast a per-minute quota."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts:
                raise
            wait = min(60, 5 * 2 ** (attempt - 1))
            log.warning("%s נכשל (ניסיון %d/%d): %s — ממתין %d שניות", what, attempt, attempts, str(e)[:200], wait)
            time.sleep(wait)


def vector_id(row):
    # stable per section+chunk, so re-running overwrites the same vector instead of adding a new one
    return f"neon-{row['number']}-{row['chunk_index']}"


def metadata(row):
    meta = {
        "section_number": row["number"],
        "chapter_title": row["chapter"],
        "page": row["page"],
        "chunk_index": row["chunk_index"],
        "version": row["version"],
        "text": row["text"],
        "source": "neon",
    }
    return {k: v for k, v in meta.items() if v is not None}  # Pinecone rejects null metadata values


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    missing = [k for k in ("DATABASE_URL", "GEMINI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX") if not getattr(config, k)]
    if missing:
        sys.exit(f"❌ חסרים ב-.env: {', '.join(missing)}")
    try:
        conn = db.connect()
        index = vectors.get_index(vectors.connect(), create=False)  # never create a new index here
    except Exception as e:
        sys.exit(f"❌ החיבור נכשל: {e}")

    with conn:
        rows = conn.execute(
            """SELECT number, chunk_index, chapter, text, page, version FROM sections
               WHERE embedded_version IS DISTINCT FROM version ORDER BY number, chunk_index"""
        ).fetchall()
        log.info("נקראו %d שורות שטרם הוטמעו", len(rows))

        upserted = skipped = 0
        for start in range(0, len(rows), BATCH_SIZE):
            batch = []
            for row in rows[start:start + BATCH_SIZE]:
                if row["text"].strip():
                    batch.append(row)
                else:
                    log.warning("דילוג על סעיף %s מקטע %s: טקסט ריק", row["number"], row["chunk_index"])
                    skipped += 1
            if not batch:
                continue
            label = f"אצווה {start // BATCH_SIZE + 1}"
            try:
                embeddings = retry(f"הטמעת {label}", lambda: vectors.embed([r["text"] for r in batch]))
                if len(embeddings) != len(batch) or any(len(e) != vectors.DIMENSION for e in embeddings):
                    raise ValueError(f"התקבלו {len(embeddings)} וקטורים בגודל לא צפוי עבור {len(batch)} שורות")
                items = [(vector_id(r), e, metadata(r)) for r, e in zip(batch, embeddings)]
                retry(f"העלאת {label} ל-Pinecone", lambda: vectors.upsert_vectors(index, items))
            except Exception as e:
                log.error("דילוג על %s (%d שורות): %s", label, len(batch), str(e)[:300])
                skipped += len(batch)
                continue
            # mark as embedded only if the row wasn't edited meanwhile; an edited row stays pending for the next run
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE sections SET embedded_version = %s WHERE number = %s AND chunk_index = %s AND version = %s",
                    [(r["version"], r["number"], r["chunk_index"], r["version"]) for r in batch],
                )
            upserted += len(batch)
            log.info("%s: הועלו %d וקטורים (סה״כ %d/%d)", label, len(batch), upserted, len(rows))

    print()
    print(f"📥 שורות שנקראו מ-Neon: {len(rows)}")
    print(f"📤 וקטורים שהועלו:      {upserted}")
    print(f"⏭️  דולגו:              {skipped}  (פירוט ב-{LOG_PATH.name})")


if __name__ == "__main__":
    main()
