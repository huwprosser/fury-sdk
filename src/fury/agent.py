from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from termcolor import cprint

from .multimodal import add_image_to_history
from .runtime import GenerationRunner, RunnerControl
from .tools import ToolExecutor, ToolRegistry
from .transport import AsyncOpenAICompatibleClient
from .types import (
    ChatStreamEvent,
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolUiEvent,
    create_tool,
)
from .utils.tts import speak_text
from .utils.voice import add_voice_message_to_history

logger = logging.getLogger(__name__)


class Agent:
    model: str
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

    def __init__(
        self,
        model: str,
        system_prompt: str,
        tools: Optional[List[Tool]] = None,
        base_url: str = "http://127.0.0.1:8080/v1",
        api_key: str = "",
        generation_params: Optional[Dict[str, Any]] = None,
        max_tool_rounds: int = 50,
        parallel_tool_calls: bool = True,
        client_options: Optional[Dict[str, Any]] = None,
        suppress_logs: bool = False,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.max_tool_rounds = max_tool_rounds
        self.stt = None
        self.tts = None
        self.base_url = base_url
        self.generation_params = generation_params or {}
        self.parallel_tool_calls = parallel_tool_calls

        self._tool_registry = ToolRegistry(tools=tools)
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

        if suppress_logs:
            self.show_yourself()

    def add_image_to_history(
        self,
        history: List[Dict[str, Any]],
        image_path: str,
    ) -> List[Dict[str, Any]]:
        return add_image_to_history(history, image_path)

    def add_voice_message_to_history(
        self,
        history: List[Dict[str, Any]],
        base64_audio_bytes: str,
    ) -> List[Dict[str, Any]]:
        return add_voice_message_to_history(history, base64_audio_bytes, self)

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

    def runner(self) -> "Runner":
        return Runner(self)

    async def chat(
        self,
        history: List[Dict[str, Any]],
        reasoning: bool = False,
        prune_unfinished_sentences: bool = False,
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        async for event in self._runner.chat(
            history=history,
            reasoning=reasoning,
            prune_unfinished_sentences=prune_unfinished_sentences,
        ):
            yield event

    async def ask_async(
        self,
        user_input: str,
        history: Optional[List[Dict[str, Any]]] = None,
        reasoning: bool = False,
        prune_unfinished_sentences: bool = False,
    ) -> str:
        return await self._runner.ask_async(
            user_input=user_input,
            history=history,
            reasoning=reasoning,
            prune_unfinished_sentences=prune_unfinished_sentences,
        )

    def ask(
        self,
        user_input: str,
        history: Optional[List[Dict[str, Any]]] = None,
        reasoning: bool = False,
        prune_unfinished_sentences: bool = False,
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
            )
        )

    def show_yourself(self) -> None:
        info = [
            ("Model", self.model),
            ("Tools", [tool["function"]["name"] for tool in self.tools]),
            (
                "System prompt",
                (self.system_prompt[:100] + "...") if self.system_prompt else None,
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
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        async for event in self._agent._runner.chat(
            history=history,
            reasoning=reasoning,
            prune_unfinished_sentences=prune_unfinished_sentences,
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
