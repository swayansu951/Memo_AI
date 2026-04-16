import ollama
import json
import re
from typing import Generator


class AGENTS:
    """Specialized agents that execute tasks based on classified intent."""

    def __init__(self, model: str = 'llama3.2:3b'):
        self.model = model

    def file_agent(self, params: dict) -> dict:
        from agent.tools import create_file
        filename = params.get("filename", "output.txt")
        content  = params.get("content", "")
        return create_file(filename, content)

    def code_agent(self, params: dict) -> dict:
        from agent.tools import write_code
        filename    = params.get("filename", "output.py")
        description = params.get("description", "")
        return write_code(filename, description, model=self.model)

    def summarizer(self, params: dict) -> dict:
        from agent.tools import summarize
        text = params.get("text", "")
        return summarize(text, model=self.model)

    def chat_agent(self, params: dict, history: list = []) -> Generator:
        from agent.tools import chat
        message = params.get("message", "")
        return chat(message, history, model=self.model)

    def run(self, intent: str, params: dict, history: list = []):
        """Route to the correct agent based on SUPERVISOR's classified intent.

        Returns:
            dict  — for create_file, write_code, summarize
            Generator[str] — for chat (stream)
        """
        dispatch = {
            "create_file":  self.file_agent,
            "write_code":   self.code_agent,
            "summarize":    self.summarizer,
            "chat":         self.chat_agent,
        }
        handler = dispatch.get(intent)
        if handler is None:
            return {
                "status":  "error",
                "action":  intent,
                "message": f"Unknown intent '{intent}'. Falling back to chat.",
            }
        if intent == "chat":
            return handler(params, history)
        return handler(params)


class SUPERVISOR:
    """Routes user text to the right agent by classifying intent via an LLM."""

    SYSTEM_PROMPT = """You are a routing supervisor. Analyze the user's message and classify it into ONE intent.

Available intents:
- "create_file"  → user wants to create a blank or simple text file / folder
- "write_code"   → user wants code generated and saved to a file
- "summarize"    → user wants content or text summarized
- "chat"         → general question, conversation, or anything else

Respond ONLY with a single valid JSON object. No markdown, no explanation — pure JSON.

JSON schema:
{
  "intent": "<one of the four intents above>",
  "params": {<intent-specific fields>}
}

Params per intent:
- create_file : {"filename": "...", "content": ""}
- write_code  : {"filename": "...", "description": "...what to code..."}
- summarize   : {"text": "...text to summarize..."}
- chat        : {"message": "...user's full message..."}

Examples:
User: "Create a Python file with a retry function"
→ {"intent":"write_code","params":{"filename":"retry.py","description":"Python function that retries a failed operation with exponential backoff"}}

User: "Make a new file called notes.txt"
→ {"intent":"create_file","params":{"filename":"notes.txt","content":""}}

User: "Summarize this: The quick brown fox jumps over the lazy dog"
→ {"intent":"summarize","params":{"text":"The quick brown fox jumps over the lazy dog"}}

User: "What is machine learning?"
→ {"intent":"chat","params":{"message":"What is machine learning?"}}
"""

    def __init__(self, user_message: str):
        self.message = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

    def main_agent(self, model: str = 'llama3.1:8b-instruct-q5_K_S') -> Generator:
        """Raw streaming generator — yields text chunks from the LLM."""
        response = ollama.chat(
            model=model,
            stream=True,
            options={"num_ctx": 4097,"num_gpu": 0, "num_thread": 8, "keep_alive": 30},
            messages=self.message,
        )
        for chunk in response:
            yield chunk["message"]["content"]

    def classify(self, model: str = 'llama3.1:8b-instruct-q5_K_S') -> dict:
        """Collect the full streamed response and parse it as a JSON intent dict.

        Returns a dict like: {"intent": "write_code", "params": {...}}
        Falls back to {"intent": "chat", ...} on any parse failure.
        """
        full_text = "".join(self.main_agent(model=model))

        # Strip markdown code fences if the LLM wraps its JSON
        cleaned = re.sub(r"```(?:json)?", "", full_text).strip().rstrip("```").strip()

        # Attempt direct JSON parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Fallback: extract the first {...} block found in the response
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Last resort: treat the entire message as a chat
        return {"intent": "chat", "params": {"message": full_text}}


class EVALUATOR:
    """Validates outputs from SUPERVISOR and AGENTS before they reach the UI."""

    VALID_INTENTS = {"create_file", "write_code", "summarize", "chat"}

    @staticmethod
    def validate_intent(intent_dict: dict) -> tuple[bool, str]:
        """Check that the SUPERVISOR returned a well-formed intent dict."""
        if not isinstance(intent_dict, dict):
            return False, "Response is not a dict"
        if "intent" not in intent_dict:
            return False, "Missing 'intent' key"
        if intent_dict["intent"] not in EVALUATOR.VALID_INTENTS:
            return False, f"Unknown intent: '{intent_dict['intent']}'"
        if "params" not in intent_dict:
            return False, "Missing 'params' key"
        return True, "OK"

    @staticmethod
    def validate_result(result) -> bool:
        """Check that a tool result dict contains a status field."""
        return isinstance(result, dict) and "status" in result