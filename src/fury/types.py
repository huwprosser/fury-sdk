from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional


class StreamError(Exception):
    """Raised when the model provider reports an error mid-stream.

    Upstream providers (e.g. routed through OpenRouter) can signal a failure
    after the HTTP response has already started by emitting a chunk whose
    ``finish_reason`` is ``"error"`` or a chunk carrying an ``error`` payload
    instead of ``choices``. Such a stream ends with partial or no output;
    surfacing it as an exception lets callers retry, fail over to another
    model, or report it — instead of silently truncating the response.
    """

    def __init__(self, message: str, *, code: Any = None) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class ToolResult:
    content: Any
    output_schema: Optional[Dict[str, Any]] = None


@dataclass
class Tool:
    name: str
    description: str
    execute: Callable[..., Any]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]


@dataclass
class HistoryDelta:
    kind: Literal[
        "assistant_tool_calls",
        "tool_result",
        "assistant_final",
        "vision_message",
    ]
    message: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ToolCallEvent:
    id: str
    tool_name: str
    status: Literal["started", "completed", "error"]
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None


@dataclass
class ToolUiEvent:
    id: str
    title: str
    type: Literal["tool_call", "other"]
    tool_call_id: Optional[str] = None
    metadata: Any = None


@dataclass
class ChatStreamEvent:
    content: Optional[str] = None
    reasoning: Optional[str] = None
    tool_call: Optional[ToolCallEvent] = None
    tool_ui: Optional[ToolUiEvent] = None
    history_delta: Optional[HistoryDelta] = None


@dataclass
class ChatResult:
    content: str
    reasoning: str
    transcript: List[Dict[str, Any]]
    tool_events: List[ToolCallEvent]
    ui_events: List[ToolUiEvent]
    interrupted: bool = False
