import asyncio
import base64
import inspect
import io
import json
import logging
import mimetypes
import re
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
)

from termcolor import cprint

from .llm_client import ChatClientProtocol, create_chat_client
from .validation import validate_history

logger = logging.getLogger(__name__)


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
class ToolCallEvent:
    tool_name: str
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None


@dataclass
class ToolUiEvent:
    id: str
    title: str
    type: Literal["tool_call", "other"]


@dataclass
class ChatStreamEvent:
    content: Optional[str] = None
    reasoning: Optional[str] = None
    tool_call: Optional[ToolCallEvent] = None
    tool_ui: Optional[ToolUiEvent] = None


def _prune_unfinished_sentences(text: str) -> str:
    """
    Prune any unfinished sentences from an input string.
    Keeps only text up to and including the last valid sentence ending punctuation.
    """
    if not text:
        return ""

    pattern = r"^(.*[.!?]+)\s*$"
    match = re.search(pattern, text, flags=re.DOTALL)

    if match:
        return match.group(1)

    return ""


def create_tool(
    id: str,
    description: str,
    execute: Callable[..., Any],
    input_schema: Dict[str, Any],
    output_schema: Dict[str, Any],
) -> Tool:
    """Create a Tool instance from the provided metadata and callbacks."""
    return Tool(
        name=id,
        description=description,
        execute=execute,
        input_schema=input_schema,
        output_schema=output_schema,
    )


class Agent:
    model: str
    system_prompt: str
    max_tool_rounds: int
    stt: Optional[Any]
    tts: Optional[Any]
    base_url: str
    history: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]
    available_functions: Dict[str, Callable[..., Any]]
    tool_objects: Dict[str, Tool]
    generation_params: Dict[str, Any]
    parallel_tool_calls: bool
    client: ChatClientProtocol

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
        suppress_logs=False,
    ) -> None:
        """
        Initialize the Agent.

        Args:
            model: The model to use.
            system_prompt: The system prompt to use.
            tools: The tools to use.
            base_url: The base URL to use.
            api_key: The API key to use.
            generation_params: The generation parameters to use.
            max_tool_rounds: The maximum number of tool rounds allowed before giving up.
            parallel_tool_calls: Whether to allow parallel tool calls.
            client_options: Extra client constructor kwargs for the HTTP client.
            suppress_logs: Prevent the Agent from logging to terminal on boot etc.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.max_tool_rounds = max_tool_rounds
        self.stt = None
        self.tts = None
        self.base_url = base_url
        self.history: List[Dict[str, Any]] = []
        self.tools: List[Dict[str, Any]] = []
        self.available_functions: Dict[str, Callable] = {}
        self.tool_objects: Dict[str, Tool] = {}
        self.generation_params = generation_params or {}
        self.parallel_tool_calls = parallel_tool_calls
        if tools:
            for tool in tools:
                if not isinstance(tool, Tool):
                    self.tools.append(tool)
                    continue

                params = tool.input_schema
                self.tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": params,
                        },
                    }
                )
                self.available_functions[tool.name] = tool.execute
                self.tool_objects[tool.name] = tool

        if self.parallel_tool_calls:
            self._register_parallel_tool()

        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})

        self.client = create_chat_client(
            base_url=base_url,
            api_key=api_key,
            **(client_options or {}),
        )

        if suppress_logs:
            self.show_yourself()

    def add_image_to_history(
        self, history: List[Dict[str, Any]], image_path: str
    ) -> List[Dict[str, Any]]:
        """Append a base64-encoded image message to the provided history."""
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        # Determine mime type
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"
        history.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            }
        )

        return history

    def add_voice_message_to_history(
        self, history: List[Dict[str, Any]], base64_audio_bytes: str
    ) -> List[Dict[str, Any]]:
        """Transcribe base64 audio and append it to the history as user content."""
        if not self.stt:
            try:
                from faster_whisper import WhisperModel
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "STT dependencies are not installed. Install fury-sdk[voice]."
                ) from exc

            self.stt = WhisperModel("base.en")

        try:
            from .utils.audio import load_audio
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Voice audio dependencies are not installed. Install fury-sdk[voice]."
            ) from exc

        audio, _ = load_audio(
            io.BytesIO(base64.b64decode(base64_audio_bytes)), sr=16000, mono=True
        )
        segments, _ = self.stt.transcribe(audio)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        print(text)
        history.append({"role": "user", "content": text})
        return history

    def speak(
        self,
        text: str,
        ref_text: str,
        ref_audio_path: Optional[str] = None,
        backbone_path: str = "neuphonic/neutts-nano-q4-gguf",
        codec_path: str = "neuphonic/neucodec-onnx-decoder",
    ) -> Any:
        """
        Generate TTS audio using the default NeuTTS backend.

        Args:
            text: Text to synthesize.
            ref_text: Reference text matching the reference audio.
            ref_audio_path: Path to a reference audio file to encode for voice conditioning.
        """

        if not ref_audio_path:
            raise ValueError("Provide ref_audio_path for TTS.")

        if not ref_text:
            raise ValueError("Provide ref_text for TTS.")

        if self.tts is None:
            try:
                from .neutts_minimal import NeuTTSMinimal
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "TTS dependencies are not installed. "
                    "Install fury-sdk[tts] and ensure espeak is available."
                ) from exc
            self.tts = NeuTTSMinimal(
                backbone_path=backbone_path,
                codec_path=codec_path,
            )

        return self.tts.infer_stream(
            text=text,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
        )

    async def chat(
        self,
        history: List[Dict[str, Any]],
        reasoning: bool = False,
        prune_unfinished_sentences: bool = False,
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        """
        Chat with the agent. Streams the response from the agent.

        Args:
            history: List of messages in the conversation.
            prune_unfinished_sentences: Whether to suppress trailing incomplete sentences
                while streaming.

        Returns:
            AsyncGenerator of ChatStreamEvent entries with content, reasoning, or tool data.
        """
        self._validate_history(history)

        try:
            active_history = self._prepare_active_history(history)
            for _ in range(self.max_tool_rounds):
                tool_calls: List[Dict[str, Any]] = []
                kwargs = self._build_chat_completion_kwargs(active_history, reasoning)
                completion = await self.client.chat.completions.create(**kwargs)
                async for event in self._stream_chat_completion_events(
                    completion=completion,
                    tool_calls=tool_calls,
                    prune_unfinished_sentences=prune_unfinished_sentences,
                ):
                    yield event

                if not tool_calls:
                    return

                active_history.append({"role": "assistant", "tool_calls": tool_calls})
                async for event in self._execute_tool_calls(
                    tool_calls=tool_calls,
                    active_history=active_history,
                ):
                    yield event

            yield ChatStreamEvent(content="Max tool rounds reached")
        except Exception as e:
            logger.exception(f"Error in chat: {e}")
            yield ChatStreamEvent(content=str(e))

    async def ask_async(
        self,
        user_input: str,
        history: Optional[List[Dict[str, Any]]] = None,
        reasoning: bool = False,
        prune_unfinished_sentences: bool = False,
    ) -> str:
        """
        Send a single user message and return the assistant response.

        Args:
            user_input: The user message to send.
            history: Optional conversation history to append to.
            reasoning: Whether to include reasoning content in the stream.
            prune_unfinished_sentences: Whether to suppress trailing incomplete sentences
                while streaming.

        Returns:
            The assistant response content as a string.
        """
        active_history = history if history is not None else []
        active_history.append({"role": "user", "content": user_input})

        buffer: List[str] = []

        self._validate_history(active_history)

        async for event in self.chat(
            active_history,
            reasoning=reasoning,
            prune_unfinished_sentences=prune_unfinished_sentences,
        ):
            if event.content:
                buffer.append(event.content)

        response = "".join(buffer)
        active_history.append({"role": "assistant", "content": response})
        return response

    def ask(
        self,
        user_input: str,
        history: Optional[List[Dict[str, Any]]] = None,
        reasoning: bool = False,
        prune_unfinished_sentences: bool = False,
    ) -> str:
        """
        Synchronous wrapper for ask_async.

        Raises:
            RuntimeError: If called from within a running event loop.
        """
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
        """Print the agent configuration summary to stdout."""
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

    def _normalize_tool_name(self, name: str) -> str:
        """Strip provider prefixes from tool names."""
        if name.startswith("functions."):
            return name.split(".", 1)[1]
        return name

    def _register_parallel_tool(self) -> None:
        """Register the built-in parallel tool wrapper."""
        tool_name = "multi_tool_use.parallel"
        if tool_name in self.available_functions:
            return

        self.available_functions[tool_name] = self._execute_parallel_tool
        self.tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "Run multiple tool calls in parallel.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool_uses": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "recipient_name": {"type": "string"},
                                        "parameters": {"type": "object"},
                                    },
                                    "required": ["recipient_name", "parameters"],
                                },
                            }
                        },
                        "required": ["tool_uses"],
                    },
                },
            }
        )

    def _filter_args(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter arguments to ensure they match the tool's input schema.
        Removes any arguments that are not defined in the schema's properties.
        """
        if tool_name not in self.tool_objects:
            return args

        tool = self.tool_objects[tool_name]
        schema = tool.input_schema

        if not isinstance(schema, dict):
            return args

        # If schema has 'properties', we filter to only allow those keys
        properties = schema.get("properties")
        if properties is not None:
            allowed_keys = set(properties.keys())
            filtered_args = {k: v for k, v in args.items() if k in allowed_keys}

            dropped_keys = set(args.keys()) - set(filtered_args.keys())
            if dropped_keys:
                logger.warning(
                    f"Dropped unexpected arguments for tool {tool_name}: {dropped_keys}"
                )

            return filtered_args

        return args

    async def _execute_parallel_tool(
        self,
        tool_uses: List[Dict[str, Any]],
        emit: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute multiple tool calls concurrently and return their results."""

        async def run_tool(tool_use: Dict[str, Any]) -> Dict[str, Any]:
            recipient_name = tool_use.get("recipient_name")
            if not recipient_name:
                return {"error": "Missing recipient_name", "tool_use": tool_use}
            params = tool_use.get("parameters") or {}
            tool_name = self._normalize_tool_name(recipient_name)
            params = self._filter_args(tool_name, params)
            if tool_name == "multi_tool_use.parallel":
                return {
                    "recipient_name": recipient_name,
                    "error": "multi_tool_use.parallel cannot invoke itself",
                }
            if tool_name not in self.available_functions:
                return {
                    "recipient_name": recipient_name,
                    "error": f"Function {tool_name} does not exist",
                }
            func = self.available_functions[tool_name]
            try:
                result = await self._invoke_tool_callable(func, params, emit)
                if isinstance(result, ToolResult):
                    result = result.content
                return {"recipient_name": recipient_name, "result": result}
            except Exception as exc:
                logger.exception(f"Error executing tool {tool_name}")
                return {
                    "recipient_name": recipient_name,
                    "error": f"Error executing {tool_name}: {exc}",
                }

        tasks = [run_tool(tool_use) for tool_use in (tool_uses or [])]
        if not tasks:
            return []
        return await asyncio.gather(*tasks)

    def _prepare_active_history(
        self, history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Copy history and ensure the system message exists when configured."""
        active_history = list(history)
        if self.system_prompt and not any(
            msg.get("role") == "system" for msg in active_history
        ):
            return [{"role": "system", "content": self.system_prompt}, *active_history]
        return active_history

    def _build_chat_completion_kwargs(
        self, active_history: List[Dict[str, Any]], reasoning: bool
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": active_history,
            "stream": True,
        }
        if not reasoning:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        if self.tools:
            kwargs["tools"] = self.tools
        if self.generation_params:
            kwargs.update(self.generation_params)
        return kwargs

    async def _stream_chat_completion_events(
        self,
        completion: Any,
        tool_calls: List[Dict[str, Any]],
        prune_unfinished_sentences: bool,
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        content_buffer = ""
        emitted_length = 0

        async for chunk in completion:  # type: ignore
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            if not delta:
                continue

            self._append_tool_call_chunks(
                tool_calls, getattr(delta, "tool_calls", None)
            )

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

    def _append_tool_call_chunks(
        self,
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

    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        active_history: List[Dict[str, Any]],
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        for tool_call in tool_calls:
            fname = tool_call["function"]["name"]
            call_id = tool_call["id"]
            missing_result = object()

            if fname not in self.available_functions:
                continue

            try:
                args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError as exc:
                payload, error_msg = self._tool_error_payload(
                    call_id,
                    fname,
                    f"Error decoding arguments for {fname}: {exc}",
                )
                yield ChatStreamEvent(content=error_msg)
                active_history.append(payload)
                continue

            filtered_args = self._filter_args(fname, args)
            yield ChatStreamEvent(
                tool_call=ToolCallEvent(
                    tool_name=fname,
                    arguments=filtered_args,
                )
            )

            result = missing_result
            async for event in self._execute_tool(fname, filtered_args):
                if (
                    event.tool_call
                    and event.tool_call.tool_name == fname
                    and event.tool_call.arguments is None
                ):
                    result = event.tool_call.result
                yield event

            normalized_result, output_schema = self._normalize_tool_result(
                None if result is missing_result else result
            )
            if output_schema:
                yield ChatStreamEvent(reasoning=f"data_{json.dumps(output_schema)}")

            tool_content, vision_message = self._format_multimodal_content(
                normalized_result
            )
            active_history.append(
                {
                    "tool_call_id": call_id,
                    "role": "tool",
                    "name": fname,
                    "content": tool_content,
                }
            )
            if vision_message:
                active_history.append(vision_message)

    def _tool_accepts_emit(self, func: Callable[..., Any]) -> bool:
        """Return True when the callable can receive a runtime-only emit callback."""
        try:
            parameters = inspect.signature(func).parameters.values()
        except (TypeError, ValueError):
            return False

        return any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            or (
                parameter.name == "emit"
                and parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            )
            for parameter in parameters
        )

    async def _invoke_tool_callable(
        self,
        func: Callable[..., Any],
        args: Dict[str, Any],
        emit: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Any:
        call_args = dict(args)
        if emit is not None and self._tool_accepts_emit(func):
            call_args["emit"] = emit

        if inspect.iscoroutinefunction(func):
            return await func(**call_args)
        return await asyncio.to_thread(func, **call_args)

    def _normalize_tool_ui_event(self, payload: Dict[str, Any]) -> ToolUiEvent:
        if not isinstance(payload, dict):
            raise TypeError("Tool UI event payload must be a dict")

        event_id = payload.get("id")
        title = payload.get("title")
        event_type = payload.get("type")

        if not isinstance(event_id, str) or not event_id:
            raise ValueError("Tool UI event payload requires a non-empty string id")
        if not isinstance(title, str) or not title:
            raise ValueError("Tool UI event payload requires a non-empty string title")
        if event_type not in ("tool_call", "other"):
            raise ValueError(
                "Tool UI event payload type must be 'tool_call' or 'other'"
            )

        return ToolUiEvent(id=event_id, title=title, type=event_type)

    async def _execute_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        func = self.available_functions[tool_name]
        queue: asyncio.Queue[Any] = asyncio.Queue()
        done = object()
        loop = asyncio.get_running_loop()

        def emit(payload: Dict[str, Any]) -> None:
            event = self._normalize_tool_ui_event(payload)
            loop.call_soon_threadsafe(queue.put_nowait, event)

        async def run_tool() -> Any:
            try:
                return await self._invoke_tool_callable(func, args, emit)
            except Exception as exc:
                logger.exception(f"Error executing tool {tool_name}")
                return f"Error: {str(exc)}"
            finally:
                loop.call_soon(queue.put_nowait, done)

        task = asyncio.create_task(run_tool())

        while True:
            item = await queue.get()
            if item is done:
                break
            yield ChatStreamEvent(tool_ui=item)

        yield ChatStreamEvent(
            tool_call=ToolCallEvent(tool_name=tool_name, result=(await task))
        )

    def _normalize_tool_result(
        self, result: Any
    ) -> tuple[Any, Optional[Dict[str, Any]]]:
        if not isinstance(result, ToolResult):
            return result, None
        return result.content, result.output_schema

    def _tool_error_payload(
        self, call_id: str, name: str, msg: str
    ) -> tuple[Dict[str, Any], str]:
        return (
            {
                "tool_call_id": call_id,
                "role": "tool",
                "name": name,
                "content": msg,
            },
            msg,
        )

    def _format_multimodal_content(
        self, result: Any
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        if not (isinstance(result, dict) and "image_base64" in result):
            return str(result), None

        description = result.get("description", "Image captured from webcam.")
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": description},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{result['image_base64']}"
                    },
                },
            ],
        }
        return description, user_message

    def _validate_history(self, history: List[Dict[str, Any]]) -> bool:
        """Validate the history is in the correct format."""
        validate_history(history)
        return True
