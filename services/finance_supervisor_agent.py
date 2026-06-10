import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:latest"


def select_agents(query: str) -> list[str]:
    """
    Supervisor Agent decides which agents
    should be invoked for the query.
    """

    prompt = f"""
You are a Finance Supervisor Agent.

Available Agents:
1. API Agent
2. Web Agent
3. RAG Agent

Rules:
- Use API Agent for stock prices, market cap and financial data.
- Use Web Agent for stock pages, URLs and company websites.
- Use RAG Agent for comparisons, analysis and reasoning.

User Query:
{query}

Return only agent names separated by commas.

Examples:

API Agent

Web Agent

API Agent,RAG Agent
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "temperature": 0
        },
        timeout=60
    )

    response.raise_for_status()

    result = response.json()["response"]

    agents = []

    if "API Agent" in result:
        agents.append("API Agent")

    if "Web Agent" in result:
        agents.append("Web Agent")

    if "RAG Agent" in result:
        agents.append("RAG Agent")

    if not agents:
        agents.append("RAG Agent")

    return agents