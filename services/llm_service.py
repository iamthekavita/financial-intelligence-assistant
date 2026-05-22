import json
from typing import Generator

import requests

# Local Ollama endpoint and default model
LLM_API_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:latest"


def _build_prompt(query: str, context: str) -> str:
    """Build a structured prompt for the LLM using the query and retrieved context."""

    # System prompt defines behavior, constraints, and output format
    system_prompt = """
You are an intelligent financial assistant.

Rules:
1. Think step-by-step before answering.
2. Show your reasoning clearly.
3. Use only the provided context.
4. Do not invent information.
5. If information is missing, say:
   "Not enough data available."

Output format:

Reasoning:
- Step 1
- Step 2
- Step 3
- Step 4

Answer:
<final answer>

Key Insights:
- insight 1
- insight 2

Confidence:
<low/medium/high>
"""

    # Final prompt passed to LLM
    return f"""
{system_prompt}

Context:
{context}

User Question:
{query}
"""


def generate_answer(query: str, context: str) -> str:
    """Generate a non-streaming answer from the local LLM endpoint."""
    prompt = _build_prompt(query, context)

    # Send request to local LLM (Ollama)
    response = requests.post(
        LLM_API_URL,
        json={
            "model": DEFAULT_MODEL,
            "messages": [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{query}
"""
            }
        ],
            
            "temperature": 0.2,
            "stream": False,
        },
        timeout=60,
    )
    response.raise_for_status()

    # Extract generated text from response JSON
    payload = response.json()
    #return payload.get("response", "No response returned from the LLM.")
    return payload["response"]


def generate_answer_stream(query: str, context: str) -> Generator[str, None, None]:
    """Stream an answer from the local LLM endpoint as the model generates it."""
    prompt = _build_prompt(query, context)

    # Enable streaming mode
    with requests.post(
        LLM_API_URL,
        json={
            "model": DEFAULT_MODEL,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": 512,
            "stream": True,
        },
        stream=True,
        timeout=60,
    ) as response:
        response.raise_for_status()

        # Process streaming chunks line-by-line
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "response" in payload:
                yield payload["response"]
