from functools import cache

from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec

import config

# ponytail: must match the embedding size; Gemini's gemini-embedding-001 gives 768 via output_dimensionality=768.
# Changing it later means deleting and recreating the index.
DIMENSION = 768
# gemini-embedding-001 returns one vector per text in a list; gemini-embedding-2 merges a list into ONE vector,
# so it can't be used for batching.
EMBED_MODEL = "gemini-embedding-001"


def connect():
    if not config.PINECONE_API_KEY or not config.PINECONE_INDEX:
        raise RuntimeError("PINECONE_API_KEY / PINECONE_INDEX missing from .env")
    return Pinecone(api_key=config.PINECONE_API_KEY)


def get_index(pc, create=True):
    """Return the index, creating it (serverless, free-tier region) if it doesn't exist yet and create=True."""
    if not pc.has_index(config.PINECONE_INDEX):
        if not create:
            raise RuntimeError(f"Index '{config.PINECONE_INDEX}' does not exist")
        pc.create_index(
            name=config.PINECONE_INDEX,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )  # blocks until the index is ready
    dim = pc.describe_index(config.PINECONE_INDEX).dimension
    if dim != DIMENSION:
        raise RuntimeError(f"Index '{config.PINECONE_INDEX}' has dimension {dim}, expected {DIMENSION}")
    return pc.Index(config.PINECONE_INDEX)


def upsert_vectors(index, items):
    """items: list of (id, values, metadata)."""
    return index.upsert(vectors=[{"id": i, "values": v, "metadata": m} for i, v, m in items], show_progress=False)


def fetch_vectors(index, ids):
    return index.fetch(ids=list(ids)).vectors


def count_vectors(index):
    return index.describe_index_stats().total_vector_count


@cache
def _genai_client():
    # kept alive for the whole run: a throwaway Client closes its connection before the request is sent
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY missing from .env")
    return genai.Client(api_key=config.GEMINI_API_KEY)


def embed(texts, task_type="RETRIEVAL_DOCUMENT"):
    """Embed up to 100 texts in one request. Use task_type="RETRIEVAL_QUERY" for search questions."""
    reply = _genai_client().models.embed_content(
        model=EMBED_MODEL,
        contents=list(texts),
        config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=DIMENSION),
    )
    return [e.values for e in reply.embeddings]
