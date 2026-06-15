from __future__ import annotations

from typing import Any, Dict, Iterable

VALID_HISTORY_ROLES = {"system", "user", "assistant", "tool"}


def validate_message(message: Dict[str, Any]) -> None:
    role = message.get("role")
    if role not in VALID_HISTORY_ROLES:
        raise ValueError(f"Invalid role: {role}")

    if role == "assistant" and isinstance(message.get("tool_calls"), list):
        return

    if message.get("content") is None:
        raise ValueError("Content is required")

    if role == "tool" and not message.get("tool_call_id"):
        raise ValueError("Tool messages require tool_call_id")


def validate_history(history: Iterable[Dict[str, Any]]) -> None:
    for message in history:
        validate_message(message)
