from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from termcolor import cprint

from .runtime import GenerationRunner, RunnerControl
from .tools import ToolExecutor, ToolRegistry
from openai import AsyncOpenAI
from .types import (
    ChatResult,
    ChatStreamEvent,
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolUiEvent,
)

logger = logging.getLogger(__name__)


class Agent:
    """Chat agent with optional tools, multimodal helpers, and streaming."""

    model: str
    base_system_prompt: str
    system_prompt: str
    max_tool_rounds: int
    base_url: str
    tools: List[Dict[str, Any]]
    available_functions: Dict[str, Any]
    tool_objects: Dict[str, Tool]
    generation_params: Dict[str, Any]
    parallel_tool_calls: bool
    auto_heal_tool_calls: bool
    client: AsyncOpenAI

    def __init__(
        self,
        model: str,
        system_prompt: str,
        tools: Optional[List[Tool]] = None,
        base_url: str = "http://127.0.0.1:8080/v1",
        api_key: str = "",
        generation_params: Optional[Dict[str, Any]] = None,
        max_tool_rounds: int = 200,
        parallel_tool_calls: bool = False,
        auto_heal_tool_calls: bool = True,
        client_options: Optional[Dict[str, Any]] = None,
        suppress_logs: bool = False,
    ) -> None:
        """Initialize an agent.

        Args:
            model: Model name sent to the OpenAI-compatible backend.
            system_prompt: Base system instruction string.
            tools: Optional list of ``Tool`` objects.
            base_url: OpenAI-compatible server base URL.
            api_key: API key for the backend.
            generation_params: Extra completion parameters such as temperature.
            max_tool_rounds: Maximum number of tool-calling rounds per request.
            parallel_tool_calls: Enable Fury's built-in parallel tool wrapper.
            auto_heal_tool_calls: Parse and execute XML-style tool calls emitted as
                assistant text by local/OpenAI-compatible models.
            client_options: Extra keyword arguments passed to the HTTP client.
            suppress_logs: Prevent printing the agent summary on initialization.
        """
        self.model = model
        self.base_system_prompt = system_prompt
        self.system_prompt = system_prompt
        self.max_tool_rounds = max_tool_rounds
        self.base_url = base_url
        self.generation_params = generation_params or {}
        self.parallel_tool_calls = parallel_tool_calls
        self.auto_heal_tool_calls = auto_heal_tool_calls
        self.suppress_logs = suppress_logs

        self._tool_registry = ToolRegistry(tools=list(tools or []))
        self._tool_executor = ToolExecutor(
            self._tool_registry,
            parallel_tool_calls=parallel_tool_calls,
            auto_heal_tool_calls=auto_heal_tool_calls,
        )
        self.tools = self._tool_registry.tools
        self.available_functions = self._tool_registry.available_functions
        self.tool_objects = self._tool_registry.tool_objects

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
            **(client_options or {}),
        )
        self._runner = GenerationRunner(
            runtime=self,
            tool_registry=self._tool_registry,
            tool_executor=self._tool_executor,
        )

        if not suppress_logs:
            self.show_yourself()

    def build_system_prompt(self) -> str:
        return self.base_system_prompt

    async def build_system_prompt_async(self) -> str:
        return self.base_system_prompt

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

    async def complete(
        self,
        history: List[Dict[str, Any]],
        reasoning: bool = False,
        prune_unfinished_sentences: bool = False,
        model: Optional[str] = None,
    ) -> ChatResult:
        return await self._agent._runner.complete(
            history=history,
            reasoning=reasoning,
            prune_unfinished_sentences=prune_unfinished_sentences,
            model=model,
            control=self._control,
        )

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
