"""
tools.py — Local tool executors for the Voice-Controlled AI Agent.

Safety rule: ALL file operations are strictly sandboxed to the output/ directory.
"""

import ollama
import re
from pathlib import Path
from typing import Generator

# ── Safety Sandbox ─────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _safe_path(filename: str) -> Path:
    """Resolve the target path and ensure it stays within output/.
    
    Raises ValueError on path-traversal attempts (e.g. '../../etc/passwd').
    """
    target = (OUTPUT_DIR / filename).resolve()
    if not str(target).startswith(str(OUTPUT_DIR.resolve())):
        raise ValueError(f"Path traversal blocked: '{filename}'")
    return target


# ── Tool: Create File ──────────────────────────────────────────────────────────
def create_file(filename: str, content: str = "") -> dict:
    """Create a file (with optional content) inside the output/ directory.

    Args:
        filename: Relative filename, e.g. 'notes.txt' or 'data/report.md'
        content:  Initial file content (empty by default)

    Returns:
        dict with keys: status, action, filepath, message
    """
    try:
        path = _safe_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "status":   "success",
            "action":   "create_file",
            "filepath": str(path),
            "message":  f"[+] File '{filename}' created in output/{filename}",
        }
    except Exception as e:
        return {"status": "error", "action": "create_file", "message": str(e)}


# ── Tool: Write Code ───────────────────────────────────────────────────────────
def write_code(
    filename: str,
    description: str,
    model: str = 'llama3.2:3b',
) -> dict:
    """Generate code from a description using the LLM and save it to output/.

    Args:
        filename:     Target file name, e.g. 'retry.py'
        description:  Natural-language description of what the code should do
        model:        Ollama model to use for generation

    Returns:
        dict with keys: status, action, filepath, code, message
    """
    try:
        prompt = (
            f"Write clean, well-commented code for the following task:\n\n"
            f"{description}\n\n"
            f"Return ONLY the code — no explanation, no markdown fences."
        )
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_ctx": 4097, "num_thread": 4, "keep_alive": "2m"},
        )
        code = response["message"]["content"].strip()

        # Strip markdown fences if the LLM added them anyway
        code = re.sub(r'^```[\w]*\n?', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n?```$', '', code, flags=re.MULTILINE)
        code = code.strip()

        path = _safe_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")

        return {
            "status":   "success",
            "action":   "write_code",
            "filepath": str(path),
            "code":     code,
            "message":  f"[+] Code written to '{filename}' in output/",
        }
    except Exception as e:
        return {"status": "error", 
                "action": "write_code", 
                "message": f"[-] Oops.. Something gone wrong: {str(e)}"
            }


# ── Tool: Summarize ────────────────────────────────────────────────────────────
def summarize(text: str, model: str = 'llama3.2:3b') -> dict:
    """Summarize the provided text using the LLM.

    Args:
        text:  The text to summarize
        model: Ollama model to use

    Returns:
        dict with keys: status, action, summary, message
    """
    try:
        if not text.strip():
            return {
                "status":  "error",
                "action":  "summarize",
                "message": "No text provided to summarize.",
            }
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role":    "system",
                    "content": "You are a precise summarizer. Summarize the text concisely in plain language.",
                },
                {"role": "user", "content": f"Summarize this:\n\n{text}"},
            ],
            options={"num_ctx": 4097, "num_thread": 4, "keep_alive": "2m"},
        )
        summary = response["message"]["content"].strip()
        return {
            "status":  "success",
            "action":  "summarize",
            "summary": summary,
            "message": "[+] Text summarized successfully.",
        }
    except Exception as e:
        return {"status": "error", "action": "summarize", "message": str(e)}


# ── Tool: Chat ─────────────────────────────────────────────────────────────────
def chat(
    message: str,
    history: list = [],
    model: str = 'llama3.2:3b',
) -> Generator[str, None, None]:
    """Stream a general chat response, including conversation history.

    Args:
        message: The current user message
        history: List of prior {"role": ..., "content": ...} messages
        model:   Ollama model to use

    Yields:
        str — text chunks as the LLM streams them
    """
    messages = [
        {"role": "system", "content": "You are a helpful, concise AI assistant."}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = ollama.chat(
        model=model,
        stream=True,
        messages=messages,
        options={"num_ctx": 4097, "num_thread": 4, "keep_alive": "2m"},
    )
    for chunk in response:
        yield chunk["message"]["content"]
