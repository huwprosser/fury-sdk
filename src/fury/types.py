from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional


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
