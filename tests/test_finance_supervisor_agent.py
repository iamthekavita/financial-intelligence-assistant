import pytest
import requests

from services.finance_supervisor_agent import select_agents


def _ollama_available() -> bool:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


@pytest.mark.skipif(not _ollama_available(), reason="Ollama is not reachable on localhost:11434")
def test_select_agents_live_returns_known_agents():
    agents = select_agents("Compare Apple and Tesla market cap")

    assert isinstance(agents, list)
    assert len(agents) > 0
    assert all(agent in {"API Agent", "Web Agent", "RAG Agent"} for agent in agents)
