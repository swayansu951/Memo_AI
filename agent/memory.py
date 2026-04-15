"""
memory.py — Session memory for the Voice-Controlled AI Agent.

Manages:
  - chat_history : list of {role, content} dicts passed to the LLM
  - action_log   : timestamped record of every pipeline run this session
"""

from datetime import datetime


class SessionMemory:
    """Holds in-memory state for a single UI session.

    Designed to be stored in Streamlit's st.session_state so it persists
    across reruns within the same browser session.
    """

    def __init__(self):
        self.chat_history: list[dict] = []   # LLM-format message history
        self.action_log:   list[dict] = []   # Human-readable pipeline log

    # ── Chat History ──────────────────────────────────────────────────────────

    def add_user_message(self, content: str) -> None:
        """Append a user turn to the LLM chat history."""
        self.chat_history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Append an assistant turn to the LLM chat history."""
        self.chat_history.append({"role": "assistant", "content": content})

    def get_history(self) -> list[dict]:
        """Return the full chat history for passing to the LLM."""
        return self.chat_history.copy()

    def clear_history(self) -> None:
        """Reset chat history (keeps action log)."""
        self.chat_history.clear()

    # ── Action Log ────────────────────────────────────────────────────────────

    def log_action(
        self,
        transcription: str,
        intent: str,
        params: dict,
        result,
    ) -> None:
        """Record one full pipeline run to the action log.

        Args:
            transcription: Raw STT output
            intent:        Classified intent string
            params:        Params returned by SUPERVISOR
            result:        Tool result dict (or accumulated chat string)
        """
        self.action_log.append(
            {
                "timestamp":     datetime.now().strftime("%H:%M:%S"),
                "transcription": transcription,
                "intent":        intent,
                "params":        params,
                "result":        result,
            }
        )

    def get_log(self) -> list[dict]:
        """Return all logged actions for this session."""
        return self.action_log.copy()

    def clear_log(self) -> None:
        """Clear the action log."""
        self.action_log.clear()

    def clear_all(self) -> None:
        """Full reset — wipes both history and log."""
        self.clear_history()
        self.clear_log()
