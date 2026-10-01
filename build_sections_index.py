"""Export every chunk (text from Neon + its vector from Pinecone) to assets/sections_index.json.

The "שאלה חופשית" tab retrieves from this one file, so dense and BM25 search exactly the same chunks.
Run after load_takanon.py + add_to_pinecone.py:  .venv/bin/python build_sections_index.py
"""
import json
import sys
from pathlib import Path

import config
import db
import vectors

OUT_PATH = Path(__file__).parent / "assets" / "sections_index.json"


def main():
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT number, chunk_index, chapter, page, text FROM sections ORDER BY string_to_array(number, '.')::int[], chunk_index"
        ).fetchall()
    index = vectors.get_index(vectors.connect(), create=False)
    ids = [f"neon-{r['number']}-{r['chunk_index']}" for r in rows]  # same ids as add_to_pinecone.py
    found = {}
    for start in range(0, len(ids), 100):
        found.update(index.fetch(ids=ids[start:start + 100]).vectors)

    missing = [i for i in ids if i not in found]
    if missing:
        sys.exit(f"❌ {len(missing)} מקטעים חסרים ב-Pinecone (הריצו add_to_pinecone.py): {missing[:5]}")

    chunks = [
        {"id": vid, "section": r["number"], "chunk_index": r["chunk_index"], "chapter": r["chapter"],
         "page": r["page"], "text": r["text"], "embedding": [round(x, 6) for x in found[vid].values]}
        for vid, r in zip(ids, rows)
    ]
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(
        {"embedding_model": vectors.EMBED_MODEL, "dimension": vectors.DIMENSION, "chunks": chunks},
        ensure_ascii=False,
    ), encoding="utf-8")
    print(f"✅ נכתבו {len(chunks)} מקטעים ל-{OUT_PATH.relative_to(OUT_PATH.parent.parent)} ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
