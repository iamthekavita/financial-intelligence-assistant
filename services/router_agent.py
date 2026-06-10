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
#         return "API Tool"

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

Choose ONLY one:

API Agent
Web Agent
RAG Agent

User Query:
{query}

Return ONLY the agent name.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "temperature": 0
        },
        timeout=120
    )

    response.raise_for_status()

    result = response.json()["response"].strip()

    # Normalize LLM output
    if "Web Agent" in result:
        return "Web Agent"

    elif "API Agent" in result:
        return "API Agent"

    elif "RAG Agent" in result:
        return "RAG Agent"

    # Safe fallback
    return "RAG Agent"