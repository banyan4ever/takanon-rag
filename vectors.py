from pinecone import Pinecone, ServerlessSpec

import config

# ponytail: must match the embedding size; Gemini's gemini-embedding-001 gives 768 via output_dimensionality=768.
# Changing it later means deleting and recreating the index.
DIMENSION = 768


def connect():
    if not config.PINECONE_API_KEY or not config.PINECONE_INDEX:
        raise RuntimeError("PINECONE_API_KEY / PINECONE_INDEX missing from .env")
    return Pinecone(api_key=config.PINECONE_API_KEY)


def get_index(pc):
    """Return the index, creating it (serverless, free-tier region) if it doesn't exist yet."""
    if not pc.has_index(config.PINECONE_INDEX):
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
