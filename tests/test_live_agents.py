import pytest
import requests

from services.embedding_service import get_embedding
from services.finance_supervisor_agent import select_agents
from services.llm_service import generate_answer, generate_answer_stream
from services.retrieval_agent import retrieve_context
from services.router_agent import route_query
from services.vector_store import VectorStore


def _ollama_available() -> bool:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


@pytest.mark.skipif(not _ollama_available(), reason="Ollama is not reachable on localhost:11434")
def test_route_query_live_returns_valid_agent():
    agent = route_query("Give me stock price for Apple")
    assert agent in {"API Agent", "Web Agent", "RAG Agent"}


@pytest.mark.skipif(not _ollama_available(), reason="Ollama is not reachable on localhost:11434")
def test_select_agents_live_returns_known_agents():
    agents = select_agents("Compare Apple and Tesla market cap")
    assert isinstance(agents, list)
    assert len(agents) > 0
    assert all(agent in {"API Agent", "Web Agent", "RAG Agent"} for agent in agents)


@pytest.mark.skipif(not _ollama_available(), reason="Ollama is not reachable on localhost:11434")
def test_generate_answer_live_non_stream():
    answer = generate_answer(
        query="What is retrieval augmented generation?",
        context="RAG means Retrieval Augmented Generation.",
        think=False,
    )
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0


@pytest.mark.skipif(not _ollama_available(), reason="Ollama is not reachable on localhost:11434")
def test_generate_answer_live_thinking_payload_shape():
    result = generate_answer(
        query="Summarize what RAG means in one line.",
        context="RAG combines retrieval with generation.",
        think=True,
    )
    assert isinstance(result, dict)
    assert "thinking" in result
    assert "response" in result


@pytest.mark.skipif(not _ollama_available(), reason="Ollama is not reachable on localhost:11434")
def test_generate_answer_stream_live_emits_chunks():
    stream = generate_answer_stream(
        query="What is RAG?",
        context="RAG stands for Retrieval Augmented Generation.",
        think=True,
    )
    chunks = list(stream)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all("thinking" in chunk and "content" in chunk for chunk in chunks)


def test_embedding_and_retrieval_live_pipeline():
    try:
        embeddings = [
            get_embedding("Apple stock information"),
            get_embedding("Tesla market capitalization details"),
            get_embedding("Microsoft company profile and valuation"),
        ]
    except Exception as exc:  # pragma: no cover - external model availability
        pytest.skip(f"Embedding model unavailable: {exc}")

    dim = len(embeddings[0])
    store = VectorStore(dim=dim)
    texts = [
        "Apple stock information",
        "Tesla market capitalization details",
        "Microsoft company profile and valuation",
    ]
    store.add(embeddings, texts)

    results = retrieve_context("What is Apple stock data?", store)

    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(item, str) for item in results)
