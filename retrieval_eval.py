"""Retrieval evaluation: dense (Pinecone) vs sparse (BM25 over Neon) against a gold query set."""
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

GOLD_PATH = Path(__file__).parent / "eval" / "gold_queries.json"
BM25_MODEL = "BM25Okapi (rank_bm25)"
TOKEN_RE = re.compile(r"\w+")
VECTOR_ID_RE = re.compile(r"neon-(.+)-\d+")


def section_id(x):
    """Accept either a section number ("10.8") or a Pinecone vector id ("neon-10.8-0")."""
    x = str(x).strip()
    m = VECTOR_ID_RE.fullmatch(x)
    return m[1] if m else x


def load_gold(text):
    """Parse and validate a gold set: a JSON list of {query, relevant_section_ids[]}. Raises ValueError."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON לא תקין: {e}")
    if not isinstance(data, list):
        raise ValueError("הקובץ חייב להכיל רשימה של שאילתות")
    gold = []
    for i, item in enumerate(data, 1):
        if not isinstance(item, dict) or not str(item.get("query", "")).strip():
            raise ValueError(f"פריט {i}: חסר שדה query")
        ids = item.get("relevant_section_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"פריט {i}: relevant_section_ids חייב להיות רשימה לא ריקה")
        gold.append({"query": item["query"].strip(), "relevant_section_ids": [section_id(x) for x in ids]})
    return gold


def score(ranked_ids, relevant, k):
    """Return (first relevant rank or None, reciprocal rank, recall) over the top k results."""
    top, relevant = ranked_ids[:k], set(relevant)
    rank = next((i for i, sid in enumerate(top, 1) if sid in relevant), None)
    return rank, (1 / rank if rank else 0.0), len(relevant & set(top)) / len(relevant)


def _unique_sections(hits):
    """Several chunks of one section count once, at the rank of its best chunk."""
    seen, out = set(), []
    for sid, s in hits:
        if sid not in seen:
            seen.add(sid)
            out.append((sid, s))
    return out


class BM25Retriever:
    # ponytail: plain \w+ tokens, no Hebrew prefix stripping (ו/ה/ב/ל/מ/ש/כ) — an honest baseline; add a stemmer to strengthen it
    def __init__(self, rows):
        self.ids = [r["number"] for r in rows]
        self.bm25 = BM25Okapi([TOKEN_RE.findall(r["text"].lower()) for r in rows])

    def search(self, query, k):
        scores = self.bm25.get_scores(TOKEN_RE.findall(query.lower()))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return _unique_sections((self.ids[i], float(scores[i])) for i in order)[:k]


def dense_search(index, vector, k):
    """Query Pinecone with a query embedding; only vectors loaded from Neon (skips test vectors)."""
    res = index.query(vector=vector, top_k=k * 2, filter={"source": "neon"}, include_metadata=True)
    return _unique_sections((m.metadata["section_number"], float(m.score)) for m in res.matches)[:k]
