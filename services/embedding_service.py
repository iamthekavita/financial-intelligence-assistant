"""Embedding service for converting text into numerical vectors."""

from sentence_transformers import SentenceTransformer

# Using a lightweight local embedding model for fast semantic similarity.
# Initialize a lightweight, local model for fast semantic embeddings.
# 'all-MiniLM-L6-v2' is commonly used for RAG due to good speed/quality tradeoff.
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> list:
    """Return an embedding vector for the provided text."""
    # model.encode returns a numpy array; convert to list for easier downstream handling
    return model.encode(text).tolist()
