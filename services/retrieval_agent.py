"""Agent responsible for semantic retrieval from FAISS."""

from services.embedding_service import get_embedding


def retrieve_context(query, vector_store):
    """
    Convert query into embedding and retrieve
    top matching chunks from FAISS.
    """

    query_embedding = get_embedding(query)

    results = vector_store.search(
        query_embedding,
        k=5
    )

    return results