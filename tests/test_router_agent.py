import pytest
import requests

from services.router_agent import route_query


def _ollama_available() -> bool:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


@pytest.mark.skipif(not _ollama_available(), reason="Ollama is not reachable on localhost:11434")
def test_route_query_live_returns_known_agent():
    result = route_query("Give me stock price for Apple")
    assert result in {"API Agent", "Web Agent", "RAG Agent"}
