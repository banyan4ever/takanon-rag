"""Free-question answering over assets/sections_index.json: the same chunks and the same prompt, two retrievers."""
import json
import time
from pathlib import Path

import numpy as np
from google.genai import types
from rank_bm25 import BM25Okapi

import vectors
from retrieval_eval import TOKEN_RE

INDEX_PATH = Path(__file__).parent / "assets" / "sections_index.json"
ANSWER_MODEL = "gemini-flash-lite-latest"
ANSWER_PROMPT = """אתה עוזר שעונה על שאלות על תקנון הלימודים של המכללה.
ענה בעברית, בקצרה ובדיוק, רק על סמך קטעי התקנון שמופיעים למטה, וציין את מספרי הסעיפים שעליהם התבססת.
אל תנחש ואל תשתמש בידע כללי. אם התשובה לא מופיעה בקטעים, השב במשפט הבא בלבד:
לא נמצא מענה לשאלה בקטעי התקנון שאוחזרו."""


def _tokens(text):
    return TOKEN_RE.findall(text.lower())


class SectionsIndex:
    def __init__(self, path=INDEX_PATH):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["embedding_model"] != vectors.EMBED_MODEL:
            raise RuntimeError(f"{path.name} built with {data['embedding_model']}, queries use {vectors.EMBED_MODEL}")
        self.chunks = data["chunks"]
        m = np.array([c["embedding"] for c in self.chunks], dtype=np.float32)
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)  # unit rows: a dot product is the cosine
        self.bm25 = BM25Okapi([_tokens(c["text"]) for c in self.chunks])

    def _top(self, scores, k):
        return [(self.chunks[i], float(scores[i])) for i in np.argsort(-scores)[:k]]

    def search_embeddings(self, question, k=3):
        q = np.array(vectors.embed([question], task_type="RETRIEVAL_QUERY")[0], dtype=np.float32)
        return self._top(self.matrix @ (q / np.linalg.norm(q)), k)

    def search_bm25(self, question, k=3):
        return self._top(self.bm25.get_scores(_tokens(question)), k)


def answer(question, hits):
    context = "\n\n".join(f"[סעיף {c['section']} | פרק: {c['chapter']}]\n{c['text']}" for c, _ in hits)
    reply = vectors._genai_client().models.generate_content(
        model=ANSWER_MODEL,
        contents=f"קטעי התקנון:\n{context}\n\nשאלה: {question}",
        config=types.GenerateContentConfig(
            system_instruction=ANSWER_PROMPT, temperature=0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    return reply.text or ""


def ask(index, question, method, k=3):
    """Retrieve with `method` ("embeddings" | "bm25") and answer. Times are in milliseconds."""
    t0 = time.perf_counter()
    hits = index.search_embeddings(question, k) if method == "embeddings" else index.search_bm25(question, k)
    t1 = time.perf_counter()
    text = answer(question, hits)
    t2 = time.perf_counter()
    return {"hits": hits, "answer": text,
            "retrieval_ms": (t1 - t0) * 1000, "answer_ms": (t2 - t1) * 1000, "total_ms": (t2 - t0) * 1000}
