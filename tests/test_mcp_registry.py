from services.mcp_registry import TOOLS


def test_tools_registry_has_required_agents():
    assert "API Agent" in TOOLS
    assert "Web Agent" in TOOLS
    assert "RAG Agent" in TOOLS

    assert callable(TOOLS["API Agent"])
    assert callable(TOOLS["Web Agent"])
    assert callable(TOOLS["RAG Agent"])
