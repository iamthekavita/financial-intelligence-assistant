"""Embedding service for converting text into vectors."""

from sentence_transformers import SentenceTransformer

# Lightweight embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def get_embedding(text: str) -> list:
    """
    Convert text into embedding vector.
    """

    return model.encode(text).tolist()