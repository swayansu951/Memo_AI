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

    SYSTEM_PROMPT = """You are a routing supervisor. Analyze the user's message and determine if it contains one or more tasks.
If there are multiple tasks (e.g., "Summarize this text and save it to a file"), you must identify all of them.

Available intents:
- "create_file"  → user wants to create a text file / folder
- "write_code"   → user wants code generated and saved to a file
- "summarize"    → user wants content or text summarized
- "chat"         → general question or conversation

Respond ONLY with a valid JSON object. No explanation.

JSON schema:
{
  "intents": [
    {
      "intent": "<intent_name>",
      "params": {<intent-specific fields>}
    },
    ...
  ]
}

Params per intent:
- create_file : {"filename": "...", "content": "..."}
- write_code  : {"filename": "...", "description": "..."}
- summarize   : {"text": "..."}
- chat        : {"message": "..."}

Examples:
User: "Summarize this: [...] and save it to results.txt"
→ {
    "intents": [
      {"intent": "summarize", "params": {"text": "[...]"}},
      {"intent": "create_file", "params": {"filename": "results.txt", "content": "SUMMARY_PLACEHOLDER"}}
    ]
  }

User: "What is AI?"
→ {
    "intents": [
      {"intent": "chat", "params": {"message": "What is AI?"}}
    ]
  }
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

        Returns a dict like: {"intents": [{"intent": "...", "params": {...}}, ...]}
        Falls back to {"intents": [{"intent": "chat", ...}]} on failure.
        """
        full_text = "".join(self.main_agent(model=model))

        cleaned = re.sub(r"```(?:json)?", "", full_text).strip().rstrip("```").strip()

        try:
            parsed = json.loads(cleaned)
            if "intents" in parsed:
                return parsed
            # Handle legacy single-intent format just in case
            if "intent" in parsed:
                return {"intents": [parsed]}
        except json.JSONDecodeError:
            pass

        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if "intents" in parsed:
                    return parsed
                if "intent" in parsed:
                    return {"intents": [parsed]}
            except json.JSONDecodeError:
                pass

        return {"intents": [{"intent": "chat", "params": {"message": full_text}}]}


class EVALUATOR:
    """Validates outputs from SUPERVISOR and AGENTS before they reach the UI."""

    VALID_INTENTS = {"create_file", "write_code", "summarize", "chat"}

    @staticmethod
    def validate_intent(intent_dict: dict) -> tuple[bool, str]:
        """Check that the SUPERVISOR returned a well-formed intents dict."""
        if not isinstance(intent_dict, dict):
            return False, "Response is not a dict"
        if "intents" not in intent_dict:
            return False, "Missing 'intents' key"
        if not isinstance(intent_dict["intents"], list):
            return False, "'intents' must be a list"
        
        for item in intent_dict["intents"]:
            if "intent" not in item:
                return False, "Item missing 'intent' key"
            if item["intent"] not in EVALUATOR.VALID_INTENTS:
                return False, f"Unknown intent: '{item['intent']}'"
            if "params" not in item:
                return False, "Item missing 'params' key"
        
        return True, "OK"

    @staticmethod
    def validate_result(result) -> bool:
        """Check that a tool result dict contains a status field."""
        return isinstance(result, dict) and "status" in result