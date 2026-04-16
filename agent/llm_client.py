"""
llm_client.py — Local Ollama LLM adapter for the Voice-Controlled AI Agent.
"""

from typing import Generator
import ollama
import httpx

def chat_complete(
    messages: list[dict],
    model: str,
    stream: bool = False,
    options: dict | None = None,
) -> dict | Generator[str, None, None]:
    kwargs: dict = {"model": model, "messages": messages}
    if options:
        kwargs["options"] = options

    error_msg = (
        "⚠️ **Error: Connection Refused.**\\n\\n"
        "Could not connect to Ollama. Please make sure the Ollama application is running on your local machine!"
    )

    if stream:
        kwargs["stream"] = True
        try:
            response = ollama.chat(**kwargs)
            def _stream_gen():
                try:
                    for chunk in response:
                        yield chunk["message"]["content"]
                except httpx.ConnectError:
                    yield error_msg
            return _stream_gen()
        except httpx.ConnectError:
            def _error_gen():
                yield error_msg
            return _error_gen()
    else:
        try:
            response = ollama.chat(**kwargs)
            return {"content": response["message"]["content"]}
        except httpx.ConnectError:
            return {"content": error_msg}
