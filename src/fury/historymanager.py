import json
from typing import Any, Dict, List, Optional

from .llm_client import ChatClientProtocol
from .validation import validate_message

DEFAULT_SUMMARY_SYSTEM_PROMPT = (
    "Summarize the conversation for future context using this format:\n"
    "## Goal\n"
    "[What the user is trying to accomplish]\n\n"
    "## Constraints & Preferences\n"
    "- [Requirements mentioned by user]\n\n"
    "## Progress\n"
    "### Done\n"
    "- [x] [Completed tasks]\n\n"
    "### In Progress\n"
    "- [ ] [Current work]\n\n"
    "### Blocked\n"
    "- [Issues, if any]\n\n"
    "## Key Decisions\n"
    "- **[Decision]**: [Rationale]\n\n"
    "## Next Steps\n"
    "1. [What should happen next]\n\n"
    "## Critical Context\n"
    "- [Data needed to continue]\n\n"
    "<read-files>\n"
    "path/to/file1\n"
    "</read-files>\n\n"
    "<modified-files>\n"
    "path/to/file2\n"
    "</modified-files>\n\n"
    "Be concise but include key decisions, filenames, commands, and TODOs."
)


class HistoryManager:
    """Manage conversation history with optional auto-compaction."""

    def __init__(
        self,
        history: Optional[List[Dict[str, Any]]] = None,
        *,
        agent: Optional["Agent"] = None,
        client: Optional[ChatClientProtocol] = None,
        summary_model: Optional[str] = None,
        auto_compact: bool = True,
        context_window: int = 32768,
        reserve_tokens: int = 8192,
        keep_recent_tokens: int = 8000,
        summary_prefix: str = "Summary of previous conversation:",
        summary_system_prompt: str = DEFAULT_SUMMARY_SYSTEM_PROMPT,
    ) -> None:
        self.history: List[Dict[str, Any]] = list(history or [])
        self.context_window = context_window
        self.reserve_tokens = reserve_tokens
        self.keep_recent_tokens = keep_recent_tokens
        self.summary_prefix = summary_prefix
        self.summary_system_prompt = summary_system_prompt
        self.auto_compact = auto_compact

        if agent is not None:
            client = client or agent.client
            summary_model = summary_model or agent.model

        self.client = client
        self.summary_model = summary_model

        if self.auto_compact and (self.client is None or self.summary_model is None):
            raise ValueError(
                "HistoryManager auto compaction requires a client and summary_model "
                "(or an Agent instance)."
            )

    async def add(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Add a message and auto-compact if configured."""
        self._validate_message(message)
        self.history.append(message)
        if self.auto_compact:
            self.history = await self._compact_history(self.history)
        return self.history

    async def extend(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Append multiple messages and auto-compact if configured."""
        for message in messages:
            self._validate_message(message)
        self.history.extend(messages)
        if self.auto_compact:
            self.history = await self._compact_history(self.history)
        return self.history

    def add_nowait(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Add a message without compacting (sync convenience)."""
        self._validate_message(message)
        self.history.append(message)
        return self.history

    def get_context_usage(self) -> tuple[int, float]:
        tokens = sum(self._estimate_tokens_for_message(msg) for msg in self.history)
        percent = (tokens / self.context_window) * 100 if self.context_window else 0.0
        return tokens, percent

    def _validate_message(self, message: Dict[str, Any]) -> None:
        validate_message(message)

    def _estimate_tokens_for_message(self, message: Dict[str, Any]) -> int:
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
                        chars += len(
                            json.dumps(block.get("arguments", {}), ensure_ascii=False)
                        )
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
            content = message.get("content", "")
            chars += len(str(content))

        return (chars + 3) // 4

    def _should_compact(self, context_tokens: int) -> bool:
        return context_tokens > self.context_window - self.reserve_tokens

    def _find_cut_index(self, messages: List[Dict[str, Any]]) -> int:
        accumulated = 0
        tentative_index = 0

        for i in range(len(messages) - 1, -1, -1):
            accumulated += self._estimate_tokens_for_message(messages[i])
            if accumulated >= self.keep_recent_tokens:
                tentative_index = i
                break
        else:
            return 0

        valid_indices = []
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role in {"user", "assistant", "system"}:
                valid_indices.append(idx)

        candidates = [idx for idx in valid_indices if idx <= tentative_index]
        if not candidates:
            return 0
        return max(candidates)

    def _extract_tool_calls(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            calls.extend([call for call in tool_calls if isinstance(call, dict)])

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

    def _parse_tool_arguments(self, arguments: Any) -> Dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                return {}
        return {}

    def _collect_file_ops(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[set[str], set[str]]:
        read_files: set[str] = set()
        modified_files: set[str] = set()

        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            for call in self._extract_tool_calls(msg):
                name = call.get("name")
                args = self._parse_tool_arguments(call.get("arguments"))
                path = args.get("path")
                if not path:
                    continue
                if name == "read":
                    read_files.add(path)
                elif name in {"edit", "write"}:
                    modified_files.add(path)

        return read_files, modified_files

    def _format_message_for_summary(self, message: Dict[str, Any]) -> str:
        role = message.get("role", "unknown")
        if role == "assistant":
            tool_calls = self._extract_tool_calls(message)
            if tool_calls:
                return (
                    f"{role} tool_calls: {json.dumps(tool_calls, ensure_ascii=False)}"
                )

        content = message.get("content", "")
        if isinstance(content, (list, dict)):
            content = json.dumps(content, ensure_ascii=False)

        if role == "tool":
            return f"tool result: {content}"

        return f"{role}: {content}"

    async def _compact_history(
        self, history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not history:
            return history

        existing_summary = None
        working_history = history

        if (
            history
            and history[0].get("role") == "system"
            and isinstance(history[0].get("content"), str)
            and history[0]["content"].startswith(self.summary_prefix)
        ):
            existing_summary = history[0]["content"][len(self.summary_prefix) :].strip()
            working_history = history[1:]

        context_tokens = sum(
            self._estimate_tokens_for_message(msg) for msg in working_history
        )
        if not self._should_compact(context_tokens):
            return history

        cut_index = self._find_cut_index(working_history)
        if cut_index <= 0:
            return history

        to_summarize = working_history[:cut_index]
        tail = working_history[cut_index:]

        lines = []
        if existing_summary:
            lines.append("Existing summary:")
            lines.append(existing_summary)

        lines.append("Conversation to summarize:")
        for msg in to_summarize:
            lines.append(self._format_message_for_summary(msg))

        read_files, modified_files = self._collect_file_ops(to_summarize)
        if read_files or modified_files:
            lines.append("")
            lines.append("Known file operations (from tool calls in this segment):")
            if read_files:
                lines.append(f"Read files: {', '.join(sorted(read_files))}")
            if modified_files:
                lines.append(f"Modified files: {', '.join(sorted(modified_files))}")

        summary_prompt = "\n".join(lines).strip()
        if not summary_prompt:
            return history

        completion = await self.client.chat.completions.create(
            model=self.summary_model,
            messages=[
                {"role": "system", "content": self.summary_system_prompt},
                {"role": "user", "content": summary_prompt},
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        summary_text = completion.choices[0].message.content or "(summary unavailable)"
        summary_message = f"{self.summary_prefix}\n{summary_text.strip()}"

        return [{"role": "system", "content": summary_message}] + tail


class StaticHistoryManager(HistoryManager):
    """Manage a fixed-size context window without summary compaction."""

    def __init__(
        self,
        *,
        target_context_length: int,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if target_context_length <= 0:
            raise ValueError("target_context_length must be greater than zero")

        super().__init__(
            history=history,
            auto_compact=False,
            context_window=target_context_length,
            reserve_tokens=0,
            keep_recent_tokens=target_context_length,
        )
        self.target_context_length = target_context_length
        self.history = self._fit_to_target(self.history)

    async def add(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._validate_message(message)
        self.history.append(message)
        self.history = self._fit_to_target(self.history)
        return self.history

    async def extend(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for message in messages:
            self._validate_message(message)
        self.history.extend(messages)
        self.history = self._fit_to_target(self.history)
        return self.history

    def add_nowait(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._validate_message(message)
        self.history.append(message)
        self.history = self._fit_to_target(self.history)
        return self.history

    def _fit_to_target(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        fitted: List[Dict[str, Any]] = []
        total_tokens = 0

        for message in reversed(messages):
            message_tokens = self._estimate_tokens_for_message(message)
            if fitted and total_tokens + message_tokens > self.target_context_length:
                break
            if not fitted and message_tokens > self.target_context_length:
                fitted = [message]
                break
            fitted.append(message)
            total_tokens += message_tokens

        return list(reversed(fitted))
