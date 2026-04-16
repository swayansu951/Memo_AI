"""
llm_client.py — Unified LLM adapter for cloud + local environments.

Priority:
  1. Groq API  (GROQ_API_KEY in env / st.secrets)  ← used on Streamlit Cloud
  2. Ollama    (localhost:11434)                     ← used locally

Model name mapping:
  Ollama names           → Groq equivalents
  llama3.1:8b-…         → llama-3.1-8b-instant
  llama3:latest          → llama3-8b-8192
  llama3-abliterated:…  → llama3-8b-8192
  mistral:latest         → mixtral-8x7b-32768
  phi3:mini              → llama-3.1-8b-instant  (closest available)
  codellama:latest       → llama3-70b-8192
"""

from __future__ import annotations
import os
from typing import Generator

# ── Groq model name mapping ────────────────────────────────────────────────────
_GROQ_MODEL_MAP: dict[str, str] = {
    # Supervisor models
    "llama3.1:8b-instruct-q5_K_S": "llama-3.1-8b-instant",
    "llama3:latest":                "llama3-8b-8192",
    "mistral:latest":               "mixtral-8x7b-32768",
    "phi3:mini":                    "llama-3.1-8b-instant",
    # Agent models
    "llama3-abliterated:latest":    "llama3-8b-8192",
    "codellama:latest":             "llama3-70b-8192",
}


def _groq_model(model: str) -> str:
    """Map an Ollama model name to its Groq equivalent (or pass through if already valid)."""
    return _GROQ_MODEL_MAP.get(model, model)


def _get_groq_key() -> str | None:
    """Try to read the Groq API key from the environment or Streamlit secrets."""
    # Direct env variable (works locally and on Streamlit Cloud)
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    # Streamlit secrets (set via the Streamlit Cloud dashboard)
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        return None


def _use_groq() -> bool:
    return bool(_get_groq_key())


# ── Public API  ────────────────────────────────────────────────────────────────

def chat_complete(
    messages: list[dict],
    model: str,
    stream: bool = False,
    options: dict | None = None,
) -> dict | Generator[str, None, None]:
    """
    Unified chat completion.

    Args:
        messages: OpenAI-style message list [{"role": ..., "content": ...}, ...]
        model:    Model name (Ollama or Groq format — will be auto-mapped)
        stream:   If True, return a generator that yields text chunks.
        options:  Ollama-style options dict (ignored for Groq).

    Returns:
        Non-streaming → dict with key "content" (the full response text)
        Streaming     → Generator[str] yielding text chunks
    """
    if _use_groq():
        return _groq_complete(messages, model, stream)
    else:
        return _ollama_complete(messages, model, stream, options)


# ── Groq backend ───────────────────────────────────────────────────────────────

def _groq_complete(
    messages: list[dict],
    model: str,
    stream: bool,
) -> dict | Generator[str, None, None]:
    from groq import Groq  # type: ignore
    client = Groq(api_key=_get_groq_key())
    groq_model = _groq_model(model)

    if stream:
        def _stream_gen():
            response = client.chat.completions.create(
                model=groq_model,
                messages=messages,
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta
        return _stream_gen()
    else:
        response = client.chat.completions.create(
            model=groq_model,
            messages=messages,
        )
        return {"content": response.choices[0].message.content or ""}


# ── Ollama backend (local only) ────────────────────────────────────────────────

def _ollama_complete(
    messages: list[dict],
    model: str,
    stream: bool,
    options: dict | None,
) -> dict | Generator[str, None, None]:
    import ollama  # type: ignore
    kwargs: dict = {"model": model, "messages": messages}
    if options:
        kwargs["options"] = options
    if stream:
        kwargs["stream"] = True
        response = ollama.chat(**kwargs)
        def _stream_gen():
            for chunk in response:
                yield chunk["message"]["content"]
        return _stream_gen()
    else:
        response = ollama.chat(**kwargs)
        return {"content": response["message"]["content"]}
