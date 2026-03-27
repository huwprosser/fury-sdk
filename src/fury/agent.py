from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from termcolor import cprint

from .memory import (
    MemorySnapshot,
    MemoryStore,
    compose_prompt_with_memory,
    create_memory_tool,
)
from .runtime import GenerationRunner, RunnerControl
from .tools import ToolExecutor, ToolRegistry
from .transport import AsyncOpenAICompatibleClient
from .types import (
    ChatStreamEvent,
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolUiEvent,
)
from .utils.tts import speak_text

logger = logging.getLogger(__name__)


def _get_registered_tool_name(tool: Any) -> Optional[str]:
    if isinstance(tool, Tool):
        return tool.name
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str):
                return name
    return None


class Agent:
    """Chat agent with optional tools, multimodal helpers, and durable memory.

    Pass ``memory_scope`` to opt into Fury's named durable-memory system.
    When enabled, the agent automatically injects that scope's memory snapshot
    into the system prompt on each turn and registers the built-in ``memory``
    tool unless disabled.
    """

    model: str
    base_system_prompt: str
    system_prompt: str
    max_tool_rounds: int
    stt: Optional[Any]
    tts: Optional[Any]
    base_url: str
    tools: List[Dict[str, Any]]
    available_functions: Dict[str, Any]
    tool_objects: Dict[str, Tool]
    generation_params: Dict[str, Any]
    parallel_tool_calls: bool
    client: AsyncOpenAICompatibleClient
    memory_store: Optional[MemoryStore]
    memory_scope: Optional[str]

    def __init__(
        self,
        model: str,
        system_prompt: str,
        tools: Optional[List[Tool]] = None,
        memory_scope: Optional[str] = None,
        memory_store: Optional[MemoryStore] = None,
        enable_memory_tool: bool = True,
        memory_tool_name: str = "memory",
        base_url: str = "http://127.0.0.1:8080/v1",
        api_key: str = "",
        generation_params: Optional[Dict[str, Any]] = None,
        max_tool_rounds: int = 200,
        parallel_tool_calls: bool = False,
        client_options: Optional[Dict[str, Any]] = None,
        suppress_logs: bool = False,
    ) -> None:
        """Initialize an agent.

        Args:
            model: Model name sent to the OpenAI-compatible backend.
            system_prompt: Base system instruction string.
            tools: Optional list of tools created with ``create_tool()``.
            memory_scope: Optional durable-memory namespace for this agent.
            memory_store: Optional store to reuse across multiple agents.
            enable_memory_tool: Register Fury's built-in scoped ``memory`` tool.
            memory_tool_name: Override the default built-in memory tool name.
            base_url: OpenAI-compatible server base URL.
            api_key: API key for the backend.
            generation_params: Extra completion parameters such as temperature.
            max_tool_rounds: Maximum number of tool-calling rounds per request.
            parallel_tool_calls: Enable Fury's built-in parallel tool wrapper.
            client_options: Extra keyword arguments passed to the HTTP client.
            suppress_logs: Prevent printing the agent summary on initialization.
        """
        self.model = model
        self.base_system_prompt = system_prompt
        self.system_prompt = system_prompt
        self.max_tool_rounds = max_tool_rounds
        self.stt = None
        self.tts = None
        self.base_url = base_url
        self.generation_params = generation_params or {}
        self.parallel_tool_calls = parallel_tool_calls
        self.memory_scope = (
            str(memory_scope).strip() if memory_scope is not None else None
        )
        self.memory_store = memory_store
        self.suppress_logs = suppress_logs

        if self.memory_scope == "":
            raise ValueError("memory_scope must be a non-empty string")
        if self.memory_scope is None and self.memory_store is not None:
            raise ValueError("memory_scope is required when memory_store is provided")
        if self.memory_scope is not None and self.memory_store is None:
            self.memory_store = MemoryStore()

        configured_tools: List[Any] = list(tools or [])
        if self.memory_scope is not None and enable_memory_tool:
            if any(
                _get_registered_tool_name(tool) == memory_tool_name
                for tool in configured_tools
            ):
                raise ValueError(
                    f"Tool name '{memory_tool_name}' is already registered."
                )
            configured_tools.append(
                create_memory_tool(
                    self.memory_store,
                    self.memory_scope,
                    name=memory_tool_name,
                )
            )

        self._tool_registry = ToolRegistry(tools=configured_tools)
        self._tool_executor = ToolExecutor(
            self._tool_registry,
            parallel_tool_calls=parallel_tool_calls,
        )
        self.tools = self._tool_registry.tools
        self.available_functions = self._tool_registry.available_functions
        self.tool_objects = self._tool_registry.tool_objects

        self.client = AsyncOpenAICompatibleClient(
            base_url=base_url,
            api_key=api_key,
            **(client_options or {}),
        )
        self._runner = GenerationRunner(
            runtime=self,
            tool_registry=self._tool_registry,
            tool_executor=self._tool_executor,
        )

        if not suppress_logs:
            self.show_yourself()

    def speak(
        self,
        text: str,
        ref_text: str,
        ref_audio_path: Optional[str] = None,
        backbone_path: str = "neuphonic/neutts-nano-q4-gguf",
        codec_path: str = "neuphonic/neucodec-onnx-decoder",
    ) -> Any:
        return speak_text(
            self,
            text=text,
            ref_text=ref_text,
            ref_audio_path=ref_audio_path,
            backbone_path=backbone_path,
            codec_path=codec_path,
        )

    def build_system_prompt(self) -> str:
        snapshot = self.capture_memory_snapshot()
        return compose_prompt_with_memory(self.base_system_prompt, snapshot)

    def capture_memory_snapshot(self) -> Optional[MemorySnapshot]:
        if self.memory_store is None or self.memory_scope is None:
            return None
        return self.memory_store.capture_snapshot(self.memory_scope)

    def runner(self) -> "Runner":
        return Runner(self)

    async def chat(
        self,
        history: List[Dict[str, Any]],
        reasoning: bool = False,
        prune_unfinished_sentences: bool = False,
        model: Optional[str] = None,
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        async for event in self._runner.chat(
            history=history,
            reasoning=reasoning,
            prune_unfinished_sentences=prune_unfinished_sentences,
            model=model,
        ):
            yield event

    async def ask_async(
        self,
        user_input: str,
        history: Optional[List[Dict[str, Any]]] = None,
        reasoning: bool = False,
        prune_unfinished_sentences: bool = False,
        model: Optional[str] = None,
    ) -> str:
        return await self._runner.ask_async(
            user_input=user_input,
            history=history,
            reasoning=reasoning,
            prune_unfinished_sentences=prune_unfinished_sentences,
            model=model,
        )

    def ask(
        self,
        user_input: str,
        history: Optional[List[Dict[str, Any]]] = None,
        reasoning: bool = False,
        prune_unfinished_sentences: bool = False,
        model: Optional[str] = None,
    ) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "Agent.ask() cannot be called from a running event loop. "
                "Use `await agent.ask_async(...)` instead."
            )

        return asyncio.run(
            self.ask_async(
                user_input=user_input,
                history=history,
                reasoning=reasoning,
                prune_unfinished_sentences=prune_unfinished_sentences,
                model=model,
            )
        )

    def show_yourself(self) -> None:
        rendered_prompt = self.build_system_prompt()
        info = [
            ("Model", self.model),
            ("Tools", [tool["function"]["name"] for tool in self.tools]),
            (
                "System prompt",
                (rendered_prompt + "...") if rendered_prompt else None,
            ),
        ]

        for label, value in info:
            cprint(f"{label}: ", "yellow", end="")
            print(value)

    async def _execute_parallel_tool(
        self,
        tool_uses: List[Dict[str, Any]],
        emit: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        return await self._tool_executor.execute_parallel_tool(tool_uses, emit=emit)


class Runner:
    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        self._control = RunnerControl()

    async def chat(
        self,
        history: List[Dict[str, Any]],
        reasoning: bool = False,
        prune_unfinished_sentences: bool = False,
        model: Optional[str] = None,
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        async for event in self._agent._runner.chat(
            history=history,
            reasoning=reasoning,
            prune_unfinished_sentences=prune_unfinished_sentences,
            model=model,
            control=self._control,
        ):
            yield event

    def interrupt(self) -> None:
        self._control.interrupt()

    def cancel(self) -> None:
        self._control.cancel()

    @property
    def interrupted(self) -> bool:
        return self._control.interrupted

    @property
    def cancelled(self) -> bool:
        return self._control.cancelled

    @property
    def partial_response(self) -> str:
        return self._control.partial_response
