import pytest

from services.api_service import fetch_all_companies
from services.mcp_registry import TOOLS
from services.processing_service import (
    api_to_dataframe,
    dataframe_to_text_chunks,
    scraped_to_dataframe,
)
from services.scraper_service import fetch_upstox_companies
from services.vector_store import VectorStore


def test_tools_registry_contains_expected_agents():
    assert "API Agent" in TOOLS
    assert "Web Agent" in TOOLS
    assert "RAG Agent" in TOOLS
    assert callable(TOOLS["API Agent"])
    assert callable(TOOLS["Web Agent"])
    assert callable(TOOLS["RAG Agent"])


@pytest.mark.asyncio
async def test_api_to_dataframe_live_data_pipeline():
    raw_data = await fetch_all_companies()

    df = api_to_dataframe(raw_data)

    assert list(df.columns) == ["company", "price", "marketCap"]

    chunks = dataframe_to_text_chunks(df, source="api")
    assert isinstance(chunks, list)

    # If API returns records, chunk format should match expected retrieval text shape.
    if chunks:
        assert "Company:" in chunks[0]
        assert "Stock Price:" in chunks[0]


def test_scraped_to_dataframe_live_data_pipeline():
    companies = fetch_upstox_companies()

    assert isinstance(companies, list)

    # The source should usually return some stock links.
    assert len(companies) > 0

    df = scraped_to_dataframe(companies)
    assert list(df.columns) == ["company", "url"]

    chunks = dataframe_to_text_chunks(df, source="web")
    assert len(chunks) == len(df)
    assert "Stock Page:" in chunks[0]


def test_vector_store_add_and_search_with_realistic_vectors():
    store = VectorStore(dim=3)

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    texts = ["apple", "microsoft", "tesla"]

    store.add(embeddings, texts)
    results = store.search([1.0, 0.0, 0.0], k=3)

    assert len(results) > 0
    assert "apple" in results
    assert all(item in texts for item in results)
