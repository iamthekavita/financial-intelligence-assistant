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
You are an intelligent financial AI assistant.

Follow these instructions carefully:

1. Analyze the question carefully.
2. Reason internally step-by-step.
3. Do expose hidden chain-of-thought.
4. Provide only concise reasoning summary.
5. Use ONLY provided context.
6. If answer is unavailable in context, say:
   "Not enough data available."

Respond in this format:

Reasoning Summary:
- concise reasoning step
- concise reasoning step

Final Answer:
<answer>

Evidence:
- <retrieved context>
"""

    # Final prompt passed to LLM
    return f"""
{system_prompt}

Context:
{context}

User Question:
{query}
"""


def generate_answer(query: str, context: str, think: bool = False):
    """Generate a non-streaming answer from the local LLM endpoint.

    If `think=True` the function will request chain-of-thought from the
    model and attempt to extract any reasoning trace included in the
    response. When thinking is enabled, the return value will be a dict with
    keys `thinking` and `response`. Otherwise a simple string is returned.
    """

    prompt = _build_prompt(query, context)

    body = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "temperature": 0.2,
        "stream": False,
    }

    if think:
        body["think"] = True

    response = requests.post(LLM_API_URL, json=body, timeout=180)
    response.raise_for_status()

    payload = response.json()

    # When thinking is requested, try to extract the trace and final answer.
    if think:
        thinking = None
        final = None

        if isinstance(payload, dict):
            thinking = payload.get("thinking")
            final = payload.get("response") or payload.get("content")

            # Some endpoints wrap results in `message`.
            if not (thinking or final) and payload.get("message"):
                msg = payload.get("message")
                if isinstance(msg, dict):
                    thinking = msg.get("thinking") or thinking
                    final = msg.get("content") or final

        return {"thinking": thinking or "", "response": final or ""}

    # Default non-thinking return is the response string
    return payload.get("response") if isinstance(payload, dict) else str(payload)


def generate_answer_stream(query: str, context: str, think: bool = True) -> Generator[dict, None, None]:
    """Stream thinking + content chunks from the local LLM endpoint.

    Yields dictionaries with keys `thinking` and `content`, where only one is
    non-None for each yielded chunk. This mirrors the Ollama `thinking` stream
    pattern and makes it easy for callers to render an interleaved trace.
    """

    prompt = _build_prompt(query, context)

    payload_body = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": True,
    }

    # Some local endpoints support an explicit `think` flag to enable
    # chain-of-thought / thinking traces. Include it when requested.
    # if think:
    #     payload_body["think"] = True

    with requests.post(
        LLM_API_URL,
        json=payload_body,
        stream=True,
        timeout=180,
    ) as response:
        response.raise_for_status()

        thinking_buffer = ""
        content_buffer = ""

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Ollama-style thinking trace: `thinking` chunks appear while the
            # model reasons, then a final `response` (or `content`) chunk
            # contains the answer. Support a few common key names.
            if "thinking" in chunk:
                text = chunk.get("thinking") or ""
                thinking_buffer += text
                yield {"thinking": text, "content": None}
            elif "response" in chunk:
                text = chunk.get("response") or ""
                content_buffer += text
                yield {"thinking": None, "content": text}
            elif "message" in chunk:
                # Some protocols wrap fields inside `message`.
                msg = chunk["message"]
                if isinstance(msg, dict):
                    if msg.get("thinking"):
                        text = msg.get("thinking")
                        thinking_buffer += text
                        yield {"thinking": text, "content": None}
                    elif msg.get("content"):
                        text = msg.get("content")
                        content_buffer += text
                        yield {"thinking": None, "content": text}
            else:
                # Unknown chunk; ignore or treat as content fallback
                text = chunk.get("text") if isinstance(chunk, dict) else None
                if text:
                    content_buffer += text
                    yield {"thinking": None, "content": text}
