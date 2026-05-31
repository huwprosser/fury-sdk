from __future__ import annotations

import json
import re
from typing import Any, Dict, List

TOOL_XML_SIGNALS = ("<tool_call>", "<function=")

_TOOL_CLOSED_PATTERNS = [
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    re.compile(r"<function=[\w-]+>.*?</function>", re.DOTALL),
]
_TOOL_ALL_PATTERNS = _TOOL_CLOSED_PATTERNS + [
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r"<function=[\w-]+>.*$", re.DOTALL),
]

_JSON_TOOL_START_RE = re.compile(r"<tool_call>\s*\{")
_FUNCTION_START_RE = re.compile(r"<function=([\w-]+)>\s*")
_TOOL_END_RE = re.compile(r"</tool_call>")
_FUNCTION_CLOSE_RE = re.compile(r"\s*</function>\s*$")
_PARAMETER_START_RE = re.compile(r"<parameter=([\w-]+)>\s*")
_PARAMETER_CLOSE_RE = re.compile(r"\s*</parameter>\s*$")


def has_tool_signal(text: str) -> bool:
    return any(signal in text for signal in TOOL_XML_SIGNALS)


def strip_tool_markup(text: str, *, final: bool = False) -> str:
    patterns = _TOOL_ALL_PATTERNS if final else _TOOL_CLOSED_PATTERNS
    for pattern in patterns:
        text = pattern.sub("", text)
    return text.strip() if final else text


def parse_tool_calls_from_text(
    content: str,
    *,
    id_offset: int = 0,
) -> List[Dict[str, Any]]:
    """Parse OpenAI-shaped tool calls from local-model XML tool markup.

    Supports both ``<tool_call>{...}`` JSON and
    ``<function=name><parameter=key>value`` forms. Closing XML tags are
    optional because small/local models frequently omit them.
    """
    tool_calls: List[Dict[str, Any]] = []

    for match in _JSON_TOOL_START_RE.finditer(content):
        json_start = match.end() - 1
        json_end = _find_json_object_end(content, json_start)
        if json_end is None:
            continue

        try:
            obj = json.loads(content[json_start : json_end + 1])
        except (json.JSONDecodeError, ValueError):
            continue

        name = obj.get("name", "")
        arguments = obj.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)

        tool_calls.append(
            {
                "id": f"call_{id_offset + len(tool_calls)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )

    if tool_calls:
        return tool_calls

    function_starts = list(_FUNCTION_START_RE.finditer(content))
    for index, match in enumerate(function_starts):
        function_name = match.group(1)
        body_start = match.end()
        next_function_start = (
            function_starts[index + 1].start()
            if index + 1 < len(function_starts)
            else len(content)
        )
        tool_end = _TOOL_END_RE.search(content[body_start:])
        body_end = (
            min(body_start + tool_end.start(), next_function_start)
            if tool_end
            else next_function_start
        )
        body = _FUNCTION_CLOSE_RE.sub("", content[body_start:body_end])
        arguments = _parse_xml_parameters(body)

        tool_calls.append(
            {
                "id": f"call_{id_offset + len(tool_calls)}",
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(arguments),
                },
            }
        )

    return tool_calls


def _find_json_object_end(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    index = start

    while index < len(text):
        char = text[index]
        if in_string:
            if char == "\\" and index + 1 < len(text):
                index += 2
                continue
            if char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1

    return None


def _parse_xml_parameters(body: str) -> Dict[str, str]:
    arguments: Dict[str, str] = {}
    parameter_starts = list(_PARAMETER_START_RE.finditer(body))

    if len(parameter_starts) == 1:
        match = parameter_starts[0]
        value = _PARAMETER_CLOSE_RE.sub("", body[match.end() :])
        arguments[match.group(1)] = value.strip()
        return arguments

    for index, match in enumerate(parameter_starts):
        value_start = match.end()
        value_end = (
            parameter_starts[index + 1].start()
            if index + 1 < len(parameter_starts)
            else len(body)
        )
        value = _PARAMETER_CLOSE_RE.sub("", body[value_start:value_end])
        arguments[match.group(1)] = value.strip()

    return arguments
