from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Protocol

import httpx

from .tools import ToolExecutor, ToolRegistry
from .types import ChatStreamEvent
from .utils.validation import validate_history

logger = logging.getLogger(__name__)


@dataclass
class RunnerControl:
    _mode: Optional[str] = None
    _partial_response: str = ""
    _stop_callback: Optional[Any] = None

    def cancel(self) -> None:
        self._request_stop("cancel")

    def interrupt(self) -> None:
        self._request_stop("interrupt")

    @property
    def cancelled(self) -> bool:
        return self._mode == "cancel"

    @property
    def interrupted(self) -> bool:
        return self._mode == "interrupt"

    @property
    def stop_requested(self) -> bool:
        return self._mode is not None

    @property
    def partial_response(self) -> str:
        return self._partial_response

    def _set_partial_response(self, text: str) -> None:
        self._partial_response = text

    def _bind_stop_callback(self, callback: Optional[Any]) -> None:
        self._stop_callback = callback

    def _request_stop(self, mode: str) -> None:
        if self._mode is None:
            self._mode = mode
        if self._stop_callback is not None:
            self._stop_callback()


class GenerationRuntime(Protocol):
    model: str
    system_prompt: str
    generation_params: Dict[str, Any]
    max_tool_rounds: int
    client: Any


class GenerationSession:
    def __init__(self, control: Optional[RunnerControl]) -> None:
        self.handle = control
        self.loop = asyncio.get_running_loop()
        self.current_stream: Optional[Any] = None
        self.current_task: Optional[asyncio.Task[Any]] = None

        if self.handle is not None:
            self.handle._bind_stop_callback(self.stop_active_work)

    @property
    def stop_requested(self) -> bool:
        return bool(self.handle and self.handle.stop_requested)

    def attach_stream(self, stream: Any) -> None:
        self.current_stream = stream

    def detach_stream(self, stream: Any) -> None:
        if self.current_stream is stream:
            self.current_stream = None

    def attach_task(self, task: asyncio.Task[Any]) -> None:
        self.current_task = task

    def detach_task(self, task: asyncio.Task[Any]) -> None:
        if self.current_task is task:
            self.current_task = None

    def update_partial_response(self, text: str) -> None:
        if self.handle is not None:
            self.handle._set_partial_response(text)

    def finalize(self, history: List[Dict[str, Any]], partial_response: str) -> None:
        if self.handle is None:
            return

        self.handle._set_partial_response(partial_response)
        self.handle._bind_stop_callback(None)
        if self.handle.interrupted and partial_response:
            history.append({"role": "assistant", "content": partial_response})

    def stop_active_work(self) -> None:
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self._stop_active_work)
            return
        self._stop_active_work()

    def _stop_active_work(self) -> None:
        if self.current_task is not None and not self.current_task.done():
            self.current_task.cancel()

        stream = self.current_stream
        if stream is None:
            return

        close = getattr(stream, "aclose", None)
        if close is None:
            return

        try:
            result = close()
        except Exception:
            return

        if asyncio.iscoroutine(result):
            self.loop.create_task(result)


def _prune_unfinished_sentences(text: str) -> str:
    if not text:
        return ""

    match = re.search(r"^(.*[.!?]+)\s*$", text, flags=re.DOTALL)
    if match:
        return match.group(1)
    return ""


def _prepare_active_history(
    history: List[Dict[str, Any]],
    system_prompt: str,
) -> List[Dict[str, Any]]:
    active_history = list(history)
    if system_prompt and not any(msg.get("role") == "system" for msg in active_history):
        return [{"role": "system", "content": system_prompt}, *active_history]
    return active_history


def _build_chat_completion_kwargs(
    *,
    model: str,
    active_history: List[Dict[str, Any]],
    reasoning: bool,
    tools: List[Dict[str, Any]],
    generation_params: Dict[str, Any],
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "messages": active_history,
        "stream": True,
    }
    if not reasoning:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        kwargs["tools"] = tools
    if generation_params:
        kwargs.update(generation_params)
    kwargs["model"] = model
    return kwargs


def _append_tool_call_chunks(
    tool_calls: List[Dict[str, Any]],
    delta_tool_calls: Optional[List[Any]],
) -> None:
    if not delta_tool_calls:
        return

    for tc_chunk in delta_tool_calls:
        if len(tool_calls) <= tc_chunk.index:
            tool_calls.append(
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            )
        tc = tool_calls[tc_chunk.index]
        if tc_chunk.id:
            tc["id"] += tc_chunk.id
        if tc_chunk.function.name:
            tc["function"]["name"] += tc_chunk.function.name
        if tc_chunk.function.arguments:
            tc["function"]["arguments"] += tc_chunk.function.arguments


def _is_expected_stop_exception(
    exc: BaseException, session: GenerationSession
) -> bool:
    if not session.stop_requested:
        return False

    return isinstance(
        exc,
        (
            asyncio.CancelledError,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.ReadError,
            httpx.RemoteProtocolError,
        ),
    )


class GenerationRunner:
    def __init__(
        self,
        *,
        runtime: GenerationRuntime,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
    ) -> None:
        self.runtime = runtime
        self.tool_registry = tool_registry
        self.tool_executor = tool_executor

    async def chat(
        self,
        history: List[Dict[str, Any]],
        reasoning: bool = False,
        prune_unfinished_sentences: bool = False,
        model: Optional[str] = None,
        control: Optional[RunnerControl] = None,
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        validate_history(history)
        response_buffer: List[str] = []
        session = GenerationSession(control)

        try:
            active_history = _prepare_active_history(history, self.runtime.system_prompt)
            for _ in range(self.runtime.max_tool_rounds):
                if session.stop_requested:
                    break

                tool_calls: List[Dict[str, Any]] = []
                kwargs = _build_chat_completion_kwargs(
                    model=model if model is not None else self.runtime.model,
                    active_history=active_history,
                    reasoning=reasoning,
                    tools=self.tool_registry.tools,
                    generation_params=self.runtime.generation_params,
                )
                completion = await self.runtime.client.chat.completions.create(**kwargs)
                session.attach_stream(completion)

                try:
                    async for event in self._stream_chat_completion_events(
                        completion=completion,
                        tool_calls=tool_calls,
                        prune_unfinished_sentences=prune_unfinished_sentences,
                    ):
                        if event.content:
                            response_buffer.append(event.content)
                            session.update_partial_response("".join(response_buffer))
                        yield event
                        if session.stop_requested:
                            break
                finally:
                    session.detach_stream(completion)
                    close = getattr(completion, "aclose", None)
                    if close is not None:
                        await close()

                if session.stop_requested:
                    break

                if not tool_calls:
                    return

                active_history.append({"role": "assistant", "tool_calls": tool_calls})
                async for event in self.tool_executor.execute_tool_calls(
                    tool_calls=tool_calls,
                    active_history=active_history,
                    session=session,
                ):
                    yield event
                    if session.stop_requested:
                        break
                if session.stop_requested:
                    break

            if not session.stop_requested:
                yield ChatStreamEvent(content="Max tool rounds reached")
        except asyncio.CancelledError as exc:
            if _is_expected_stop_exception(exc, session):
                return
            raise
        except Exception as exc:
            if _is_expected_stop_exception(exc, session):
                return
            logger.exception(f"Error in chat: {exc}")
            yield ChatStreamEvent(content=str(exc))
        finally:
            session.finalize(history, "".join(response_buffer))

    async def ask_async(
        self,
        user_input: str,
        history: Optional[List[Dict[str, Any]]] = None,
        reasoning: bool = False,
        prune_unfinished_sentences: bool = False,
        model: Optional[str] = None,
    ) -> str:
        active_history = history if history is not None else []
        active_history.append({"role": "user", "content": user_input})

        validate_history(active_history)

        buffer: List[str] = []
        async for event in self.chat(
            active_history,
            reasoning=reasoning,
            prune_unfinished_sentences=prune_unfinished_sentences,
            model=model,
        ):
            if event.content:
                buffer.append(event.content)

        response = "".join(buffer)
        active_history.append({"role": "assistant", "content": response})
        return response

    async def _stream_chat_completion_events(
        self,
        completion: Any,
        tool_calls: List[Dict[str, Any]],
        prune_unfinished_sentences: bool,
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        content_buffer = ""
        emitted_length = 0

        async for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            if not delta:
                continue

            _append_tool_call_chunks(tool_calls, getattr(delta, "tool_calls", None))

            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content:
                yield ChatStreamEvent(reasoning=reasoning_content)
                continue

            if not delta.content:
                continue

            if prune_unfinished_sentences:
                content_buffer += delta.content
                pruned = _prune_unfinished_sentences(content_buffer)
                if len(pruned) <= emitted_length:
                    continue
                fresh = pruned[emitted_length:]
                emitted_length = len(pruned)
                yield ChatStreamEvent(content=fresh)
                continue

            yield ChatStreamEvent(content=delta.content)
