import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .multimodal import build_image_message
from .transport import AsyncOpenAICompatibleClient
from .utils.history_summary import (
    build_summary_prompt,
    estimate_message_tokens,
    find_history_cut_index,
)
from .utils.validation import validate_message
from .utils.voice import add_voice_message_to_history, prewarm_transcription_model

if TYPE_CHECKING:
    from .agent import Agent

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
DEFAULT_HISTORY_ROOT = ".fury/history"


def _looks_like_agent(value: Any) -> bool:
    return (
        value is not None
        and hasattr(value, "runner")
        and hasattr(value, "client")
        and hasattr(value, "model")
    )


class _ManagedHistory(list):
    def __init__(
        self,
        owner: "HistoryManager",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(messages or [])
        self._owner = owner

    def append(self, message: Dict[str, Any]) -> None:
        self._owner._validate_message(message)
        self._owner._append_persisted_messages([message])
        super().append(message)

    def extend(self, messages: List[Dict[str, Any]]) -> None:
        pending = list(messages)
        for message in pending:
            self._owner._validate_message(message)
        self._owner._append_persisted_messages(pending)
        super().extend(pending)


class HistoryManager:
    """Manage conversation history with optional auto-compaction."""

    def __init__(
        self,
        history: Optional[List[Dict[str, Any]]] = None,
        *,
        agent: Optional["Agent"] = None,
        client: Optional[AsyncOpenAICompatibleClient] = None,
        summary_model: Optional[str] = None,
        auto_compact: bool = True,
        context_window: int = 32768,
        reserve_tokens: int = 8192,
        keep_recent_tokens: int = 8000,
        summary_prefix: str = "Summary of previous conversation:",
        summary_system_prompt: str = DEFAULT_SUMMARY_SYSTEM_PROMPT,
        persist_to_disk: bool = False,
        session_id: Optional[str] = None,
        history_root: str = DEFAULT_HISTORY_ROOT,
        show_compaction_status: Optional[bool] = None,
    ) -> None:
        initial_history, agent = self._coerce_history_and_agent(history, agent)
        self._disk_lock = RLock()
        self.agent = agent
        self.context_window = context_window
        self.reserve_tokens = reserve_tokens
        self.keep_recent_tokens = keep_recent_tokens
        self.summary_prefix = summary_prefix
        self.summary_system_prompt = summary_system_prompt
        self.auto_compact = auto_compact
        self.persist_to_disk = persist_to_disk
        self.session_id = str(session_id).strip() if session_id is not None else None
        self.history_root = Path(history_root)
        self.history_path: Optional[Path] = None
        self._needs_load_compaction = False
        if show_compaction_status is None:
            show_compaction_status = not bool(getattr(agent, "suppress_logs", False))
        self.show_compaction_status = bool(show_compaction_status)

        if agent is not None:
            client = client or agent.client
            summary_model = summary_model or agent.model
            prewarm_transcription_model(agent)

        self.client = client
        self.summary_model = summary_model

        if self.auto_compact and (self.client is None or self.summary_model is None):
            raise ValueError(
                "HistoryManager auto compaction requires a client and summary_model "
                "(or an Agent instance)."
            )

        if self.persist_to_disk:
            if not self.session_id:
                raise ValueError(
                    "session_id is required when persist_to_disk is enabled."
                )
            self.history_path = self._path_for_session(self.session_id)
            loaded_history = self._load_persisted_history()
            if loaded_history:
                self._set_history(loaded_history)
                self._needs_load_compaction = self.auto_compact
            else:
                self._set_history(initial_history)
                if self.history:
                    self._rewrite_persisted_history(list(self.history))
        else:
            self._set_history(initial_history)

    async def add(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Add a message and auto-compact if configured."""
        await self._compact_loaded_history_if_needed()
        self.history.append(message)
        if self.auto_compact:
            self._set_history(await self._compact_history(list(self.history)))
        return self.history

    async def extend(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Append multiple messages and auto-compact if configured."""
        await self._compact_loaded_history_if_needed()
        self.history.extend(messages)
        if self.auto_compact:
            self._set_history(await self._compact_history(list(self.history)))
        return self.history

    def add_nowait(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Add a message without compacting (sync convenience)."""
        self.history.append(message)
        return self.history

    async def add_image(
        self,
        image_path: str,
        *,
        text: str = "Image input.",
    ) -> List[Dict[str, Any]]:
        return await self.add(build_image_message(image_path, text=text))

    async def add_voice(self, base64_audio_bytes: str) -> List[Dict[str, Any]]:
        if self.agent is None:
            raise ValueError("HistoryManager.add_voice() requires an Agent instance.")
        message = add_voice_message_to_history(
            [],
            base64_audio_bytes,
            self.agent,
        )[0]
        return await self.add(message)

    def get_context_usage(self) -> tuple[int, float]:
        tokens = sum(estimate_message_tokens(msg) for msg in self.history)
        percent = (tokens / self.context_window) * 100 if self.context_window else 0.0
        return tokens, percent

    def _validate_message(self, message: Dict[str, Any]) -> None:
        validate_message(message)

    def _set_history(self, messages: List[Dict[str, Any]]) -> None:
        self.history = _ManagedHistory(self, messages)

    async def _compact_loaded_history_if_needed(self) -> None:
        if not self._needs_load_compaction:
            return
        self._set_history(
            await self._compact_history(
                list(self.history),
                context_label="restored history",
            )
        )
        self._needs_load_compaction = False

    def _print_compaction_status(self, context_label: str, context_tokens: int) -> None:
        if not self.show_compaction_status:
            return
        print(
            f"[history] Compacting {context_label} "
            f"(estimated {context_tokens} tokens)...",
            flush=True,
        )

    @staticmethod
    def _coerce_history_and_agent(
        history: Optional[List[Dict[str, Any]]],
        agent: Optional["Agent"],
    ) -> tuple[List[Dict[str, Any]], Optional["Agent"]]:
        if agent is None and _looks_like_agent(history):
            return [], history
        return list(history or []), agent

    def _estimate_tokens_for_message(self, message: Dict[str, Any]) -> int:
        return estimate_message_tokens(message)

    def _path_for_session(self, session_id: str) -> Path:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", session_id).strip("-")
        digest = hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:10]
        filename = f"{slug or 'session'}-{digest}.jsonl"
        return self.history_root / filename

    def _load_persisted_history(self) -> List[Dict[str, Any]]:
        if self.history_path is None or not self.history_path.exists():
            return []

        history: List[Dict[str, Any]] = []
        with self._disk_lock:
            try:
                with open(self.history_path, "r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(payload, dict):
                            continue
                        try:
                            validate_message(payload)
                        except ValueError:
                            continue
                        history.append(payload)
            except OSError:
                return []
        return history

    def _serialize_message(self, message: Dict[str, Any]) -> str:
        try:
            return json.dumps(message, ensure_ascii=False)
        except TypeError as exc:
            raise ValueError(
                "History messages must be JSON serializable when persist_to_disk is enabled."
            ) from exc

    def _append_persisted_messages(self, messages: List[Dict[str, Any]]) -> None:
        if not self.persist_to_disk or self.history_path is None or not messages:
            return

        serialized_messages = [
            self._serialize_message(message)
            for message in messages
        ]
        path = self.history_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._disk_lock:
            with open(path, "a", encoding="utf-8") as handle:
                for serialized in serialized_messages:
                    handle.write(serialized + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def _rewrite_persisted_history(self, messages: List[Dict[str, Any]]) -> None:
        if not self.persist_to_disk or self.history_path is None:
            return

        serialized_messages = [
            self._serialize_message(message)
            for message in messages
        ]
        path = self.history_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix=".history-",
            suffix=".jsonl",
            dir=str(path.parent),
        )
        try:
            with self._disk_lock:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    for serialized in serialized_messages:
                        handle.write(serialized + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _should_compact(self, context_tokens: int) -> bool:
        return context_tokens > self.context_window - self.reserve_tokens

    def _find_cut_index(self, messages: List[Dict[str, Any]]) -> int:
        return find_history_cut_index(
            messages,
            keep_recent_tokens=self.keep_recent_tokens,
        )

    async def _compact_history(
        self,
        history: List[Dict[str, Any]],
        *,
        context_label: str = "history",
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

        context_tokens = sum(estimate_message_tokens(msg) for msg in working_history)
        if not self._should_compact(context_tokens):
            return history

        cut_index = self._find_cut_index(working_history)
        if cut_index <= 0:
            return history

        to_summarize = working_history[:cut_index]
        tail = working_history[cut_index:]

        summary_prompt = build_summary_prompt(
            to_summarize,
            existing_summary=existing_summary,
        )
        if not summary_prompt:
            return history

        self._print_compaction_status(context_label, context_tokens)

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
        self._set_history(self._fit_to_target(list(self.history)))

    async def add(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.history.append(message)
        self._set_history(self._fit_to_target(list(self.history)))
        return self.history

    async def extend(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.history.extend(messages)
        self._set_history(self._fit_to_target(list(self.history)))
        return self.history

    def add_nowait(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.history.append(message)
        self._set_history(self._fit_to_target(list(self.history)))
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
