# """Agent responsible for routing user queries."""


# def route_query(query: str) -> str:

#     query = query.lower()

#     # Web-related queries
#     web_keywords = [
#         "stock page",
#         "upstox",
#         "url",
#         "website"
#     ]

#     # API / financial info queries
#     api_keywords = [
#         "price",
#         "market cap",
#         "ceo",
#         "revenue",
#         "financial",
#         "company",
#         "stock",
#         "tesla",
#         "apple",
#         "google",
#         "microsoft"
#     ]

#     # Check web queries
#     if any(word in query for word in web_keywords):
#         return "Web Agent"

#     # Check financial/API queries
#     if any(word in query for word in api_keywords):
#         return "API Agent"

#     # Default semantic search
#     return "RAG Agent"

"""LLM-based Router Agent."""

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:latest"


def route_query(query: str) -> str:
    """
    Use LLM reasoning to decide which agent
    should handle the user query.
    """

    prompt = f"""
You are a routing agent.

Available agents:
1. API Agent
2. Web Agent
3. RAG Agent

Rules:
- API Agent handles financial/company data queries.
- Web Agent handles URL, stock page, or website queries.
- RAG Agent handles semantic search, comparisons, or general reasoning.

User Query:
{query}

Return ONLY one of:
API Agent
Web Agent
RAG Agent
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "temperature": 0
        },
        timeout=30
    )

    response.raise_for_status()

    result = response.json()["response"].strip()

    return result