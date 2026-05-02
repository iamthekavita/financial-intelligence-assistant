import json
from typing import Generator

import requests

# Local Ollama endpoint and default model
LLM_API_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"


def _build_prompt(query: str, context: str) -> str:
    """Build a structured prompt for the LLM using the query and retrieved context."""

    # System prompt defines behavior, constraints, and output format
    system_prompt = """
You are an intelligent financial assistant.

Follow these rules EXACTLY:
1. Think carefully and show your step-by-step reasoning.
2. Do NOT hide your reasoning or chain-of-thought.
3. Use ONLY the provided context.
4. If the answer is not directly supported by the context, respond exactly:
Not enough data available.

Respond in this exact format and nothing else:
Reasoning:
- <step 1>
- <step 2>
- ...

Answer:
<final answer>

Key Insights:
- <point 1>
- <point 2>

Data Used:
- <relevant data from context>
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
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": 512,
            "stream": False,
        },
        timeout=60,
    )
    response.raise_for_status()

    # Extract generated text from response JSON
    payload = response.json()
    return payload.get("response", "No response returned from the LLM.")


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
