from __future__ import annotations

import json
from typing import Any, Dict, List


def estimate_message_tokens(message: Dict[str, Any]) -> int:
    role = message.get("role")
    chars = 0

    if role in {"user", "system"}:
        content = message.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    chars += len(block.get("text", ""))
        elif isinstance(content, dict):
            chars += len(json.dumps(content, ensure_ascii=False))
        else:
            chars += len(str(content))
    elif role == "assistant":
        content = message.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    chars += len(block.get("text", ""))
                elif isinstance(block, dict) and block.get("type") == "toolCall":
                    chars += len(block.get("name", ""))
                    chars += len(json.dumps(block.get("arguments", {}), ensure_ascii=False))
        else:
            chars += len(str(content))

        tool_calls = message.get("tool_calls")
        if tool_calls:
            chars += len(json.dumps(tool_calls, ensure_ascii=False))
    elif role == "tool":
        content = message.get("content", "")
        if isinstance(content, (dict, list)):
            chars += len(json.dumps(content, ensure_ascii=False))
        else:
            chars += len(str(content))
    else:
        chars += len(str(message.get("content", "")))

    return (chars + 3) // 4


def find_history_cut_index(
    messages: List[Dict[str, Any]],
    *,
    keep_recent_tokens: int,
) -> int:
    accumulated = 0
    tentative_index = 0

    for i in range(len(messages) - 1, -1, -1):
        accumulated += estimate_message_tokens(messages[i])
        if accumulated >= keep_recent_tokens:
            tentative_index = i
            break
    else:
        return 0

    candidates = [
        idx
        for idx, msg in enumerate(messages)
        if msg.get("role") in {"user", "assistant", "system"} and idx <= tentative_index
    ]
    if not candidates:
        return 0
    return max(candidates)


def build_summary_prompt(
    messages: List[Dict[str, Any]],
    *,
    existing_summary: str | None = None,
) -> str:
    lines = []
    if existing_summary:
        lines.append("Existing summary:")
        lines.append(existing_summary)

    lines.append("Conversation to summarize:")
    for msg in messages:
        lines.append(_format_message_for_summary(msg))

    read_files, modified_files = _collect_file_ops(messages)
    if read_files or modified_files:
        lines.append("")
        lines.append("Known file operations (from tool calls in this segment):")
        if read_files:
            lines.append(f"Read files: {', '.join(sorted(read_files))}")
        if modified_files:
            lines.append(f"Modified files: {', '.join(sorted(modified_files))}")

    return "\n".join(lines).strip()


def _format_message_for_summary(message: Dict[str, Any]) -> str:
    role = message.get("role", "unknown")
    if role == "assistant":
        tool_calls = _extract_tool_calls(message)
        if tool_calls:
            return f"{role} tool_calls: {json.dumps(tool_calls, ensure_ascii=False)}"

    content = message.get("content", "")
    if isinstance(content, (list, dict)):
        content = json.dumps(content, ensure_ascii=False)

    if role == "tool":
        return f"tool result: {content}"
    return f"{role}: {content}"


def _collect_file_ops(messages: List[Dict[str, Any]]) -> tuple[set[str], set[str]]:
    read_files: set[str] = set()
    modified_files: set[str] = set()

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in _extract_tool_calls(msg):
            path = _parse_tool_arguments(call.get("arguments")).get("path")
            if not path:
                continue
            if call.get("name") == "read":
                read_files.add(path)
            elif call.get("name") in {"edit", "write"}:
                modified_files.add(path)

    return read_files, modified_files


def _extract_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        calls.extend(call for call in tool_calls if isinstance(call, dict))

    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "toolCall":
                calls.append(
                    {
                        "name": block.get("name"),
                        "arguments": block.get("arguments"),
                    }
                )
    return calls


def _parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    return {}
