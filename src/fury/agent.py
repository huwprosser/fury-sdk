import io
import json
import logging
import inspect
import base64
import mimetypes
import requests
import asyncio
import re
from termcolor import cprint
from .utils.audio import load_audio
from dataclasses import dataclass
from typing import (
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Any,
    Union,
)
from openai import AsyncOpenAI

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
    announcement_phrase: Union[str, Callable[[Dict[str, Any]], str]]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]


@dataclass
class ToolCallEvent:
    tool_name: str
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    announcement_phrase: Optional[str] = None


@dataclass
class ChatStreamEvent:
    content: Optional[str] = None
    reasoning: Optional[str] = None
    tool_call: Optional[ToolCallEvent] = None


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
    announcement_phrase: Union[str, Callable[[Dict[str, Any]], str]],
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
        announcement_phrase=announcement_phrase,
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
    client: AsyncOpenAI

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

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
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
            import whisper

            self.stt = whisper.load_model("base.en")

        audio, _ = load_audio(
            io.BytesIO(base64.b64decode(base64_audio_bytes)), sr=16000, mono=True
        )
        result = self.stt.transcribe(audio)
        print(result["text"])
        history.append({"role": "user", "content": result["text"]})
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

        if not self._validate_history(history):
            raise ValueError("History is not in the correct format")

        def format_multimodal_content(res):
            if isinstance(res, dict) and "image_base64" in res:
                description = res.get("description", "Image captured from webcam.")
                user_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": description,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{res['image_base64']}"
                            },
                        },
                    ],
                }
                return description, user_message
            return str(res), None

        try:

            if history is not None:
                # Ensure the system prompt is included when caller provides history.
                active_history = list(history)
                if self.system_prompt and not any(
                    msg.get("role") == "system" for msg in active_history
                ):
                    active_history = [
                        {"role": "system", "content": self.system_prompt},
                        *active_history,
                    ]

            def tool_error_payload(call_id: str, name: str, msg: str):
                return (
                    {
                        "tool_call_id": call_id,
                        "role": "tool",
                        "name": name,
                        "content": msg,
                    },
                    msg,
                )

            for _ in range(self.max_tool_rounds):
                tool_calls: List[Dict[str, Any]] = []
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": active_history,
                    "stream": True,
                }
                if not reasoning:
                    kwargs["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": False}
                    }
                if self.tools:
                    kwargs["tools"] = self.tools

                if self.generation_params:
                    kwargs.update(self.generation_params)

                completion = await self.client.chat.completions.create(**kwargs)

                content_buffer = ""
                emitted_length = 0

                async for chunk in completion:  # type: ignore
                    delta = chunk.choices[0].delta
                    if delta.tool_calls:
                        for tc_chunk in delta.tool_calls:
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
                                tc["function"][
                                    "arguments"
                                ] += tc_chunk.function.arguments

                    reasoning_content = getattr(delta, "reasoning_content", None)
                    if reasoning_content:
                        yield ChatStreamEvent(reasoning=reasoning_content)
                        continue

                    if delta.content:
                        if prune_unfinished_sentences:
                            content_buffer += delta.content
                            pruned = _prune_unfinished_sentences(content_buffer)
                            if len(pruned) > emitted_length:
                                fresh = pruned[emitted_length:]
                                emitted_length = len(pruned)
                                yield ChatStreamEvent(content=fresh)
                        else:
                            yield ChatStreamEvent(content=delta.content)

                if not tool_calls:
                    return

                active_history.append({"role": "assistant", "tool_calls": tool_calls})

                for tool_call in tool_calls:
                    fname = tool_call["function"]["name"]
                    call_id = tool_call["id"]

                    if fname not in self.available_functions:
                        continue

                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError as e:
                        payload, error_msg = tool_error_payload(
                            call_id, fname, f"Error decoding arguments for {fname}: {e}"
                        )
                        yield ChatStreamEvent(content=error_msg)
                        active_history.append(payload)
                        continue

                    try:
                        args = self._filter_args(fname, args)
                        announcement = self._build_tool_announcement_phrase(fname, args)
                        yield ChatStreamEvent(
                            tool_call=ToolCallEvent(
                                tool_name=fname,
                                arguments=args,
                                announcement_phrase=announcement,
                            )
                        )

                        func = self.available_functions[fname]
                        result = (
                            await func(**args)
                            if inspect.iscoroutinefunction(func)
                            else func(**args)
                        )
                    except Exception as e:
                        logger.exception(f"Error executing tool {fname}")
                        result = f"Error: {str(e)}"

                    if isinstance(result, ToolResult):
                        if result.output_schema:
                            yield ChatStreamEvent(
                                reasoning=f"data_{json.dumps(result.output_schema)}"
                            )
                        result = result.content

                    yield ChatStreamEvent(
                        tool_call=ToolCallEvent(tool_name=fname, result=result)
                    )

                    tool_content, vision_message = format_multimodal_content(result)
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

        if not self._validate_history(active_history):
            raise ValueError("History is not in the correct format")

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

    def _build_tool_announcement_phrase(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> str:
        """Create a human-readable announcement for a tool call."""
        if tool_name in self.tool_objects:
            phrase = str(self.tool_objects[tool_name].announcement_phrase)
            return f"Using {phrase.replace('[args]', str(arguments))}..."

        return f"Using {tool_name}..."

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
        self, tool_uses: List[Dict[str, Any]]
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
                if inspect.iscoroutinefunction(func):
                    result = await func(**params)
                else:
                    result = await asyncio.to_thread(func, **params)
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

    def _validate_history(self, history: List[Dict[str, Any]]) -> bool:
        """Validate the history is in the correct format."""
        for message in history:
            if message.get("role") not in ["system", "user", "assistant", "tool"]:
                raise ValueError(f"Invalid role: {message.get('role')}")
            if message.get("content") is None:
                raise ValueError("Content is required")

        return True
