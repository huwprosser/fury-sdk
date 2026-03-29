from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from .types import ChatStreamEvent, Tool, ToolCallEvent, ToolResult, ToolUiEvent

logger = logging.getLogger(__name__)


def _normalize_tool_name(name: str) -> str:
    if name.startswith("functions."):
        return name.split(".", 1)[1]
    return name


def _normalize_tool_ui_event(payload: Dict[str, Any]) -> ToolUiEvent:
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
        raise ValueError("Tool UI event payload type must be 'tool_call' or 'other'")

    return ToolUiEvent(id=event_id, title=title, type=event_type)


def _normalize_tool_result(result: Any) -> tuple[Any, Optional[Dict[str, Any]]]:
    if not isinstance(result, ToolResult):
        return result, None
    return result.content, result.output_schema


def _tool_error_payload(call_id: str, name: str, msg: str) -> tuple[Dict[str, Any], str]:
    return (
        {
            "tool_call_id": call_id,
            "role": "tool",
            "name": name,
            "content": msg,
        },
        msg,
    )


def _format_multimodal_content(result: Any) -> tuple[str, Optional[Dict[str, Any]]]:
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


class ToolRegistry:
    def __init__(self, tools: Optional[List[Tool]] = None) -> None:
        self.tools: List[Dict[str, Any]] = []
        self.available_functions: Dict[str, Callable[..., Any]] = {}
        self.tool_objects: Dict[str, Tool] = {}

        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool | Dict[str, Any]) -> None:
        if not isinstance(tool, Tool):
            self.tools.append(tool)
            return

        self.tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
        )
        self.available_functions[tool.name] = tool.execute
        self.tool_objects[tool.name] = tool

    def register_builtin(
        self,
        *,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable[..., Any],
    ) -> None:
        if name in self.available_functions:
            return

        self.available_functions[name] = func
        self.tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )

    def filter_args(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.tool_objects:
            return args

        schema = self.tool_objects[tool_name].input_schema
        if not isinstance(schema, dict):
            return args

        properties = schema.get("properties")
        if properties is None:
            return args

        allowed_keys = set(properties.keys())
        filtered_args = {k: v for k, v in args.items() if k in allowed_keys}

        dropped_keys = set(args.keys()) - set(filtered_args.keys())
        if dropped_keys:
            logger.warning(
                f"Dropped unexpected arguments for tool {tool_name}: {dropped_keys}"
            )

        return filtered_args


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        parallel_tool_calls: bool = True,
    ) -> None:
        self.registry = registry
        if parallel_tool_calls:
            self._register_parallel_tool()

    async def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        active_history: List[Dict[str, Any]],
        session: Optional[Any] = None,
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        for tool_call in tool_calls:
            if session and session.stop_requested:
                return

            fname = tool_call["function"]["name"]
            call_id = tool_call["id"]
            missing_result = object()

            if fname not in self.registry.available_functions:
                payload, error_msg = _tool_error_payload(
                    call_id,
                    fname,
                    f"Error: tool '{fname}' is not registered.",
                )
                yield ChatStreamEvent(
                    tool_call=ToolCallEvent(tool_name=fname, result=error_msg)
                )
                active_history.append(payload)
                continue

            try:
                args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError as exc:
                payload, error_msg = _tool_error_payload(
                    call_id,
                    fname,
                    f"Error decoding arguments for {fname}: {exc}",
                )
                yield ChatStreamEvent(content=error_msg)
                active_history.append(payload)
                continue

            filtered_args = self.registry.filter_args(fname, args)
            yield ChatStreamEvent(
                tool_call=ToolCallEvent(tool_name=fname, arguments=filtered_args)
            )

            result = missing_result
            tool_error = None
            try:
                async for event in self.execute_tool(fname, filtered_args, session=session):
                    if (
                        event.tool_call
                        and event.tool_call.tool_name == fname
                        and event.tool_call.arguments is None
                    ):
                        result = event.tool_call.result
                    yield event
                    if session and session.stop_requested:
                        return
            except Exception as exc:
                # Capture the error and yield it as a tool result
                tool_error = f"Error executing {fname}: {str(exc)}"
                logger.warning(tool_error)
            
            # Check if there was an error during tool execution
            if tool_error:
                # Yield the error as a tool result
                yield ChatStreamEvent(
                    tool_call=ToolCallEvent(
                        tool_name=fname, 
                        result=tool_error
                    )
                )
                normalized_result = tool_error
            else:
                normalized_result, output_schema = _normalize_tool_result(
                    None if result is missing_result else result
                )
                if output_schema:
                    yield ChatStreamEvent(reasoning=f"data_{json.dumps(output_schema)}")

            tool_content, vision_message = _format_multimodal_content(normalized_result)
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

    async def execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        session: Optional[Any] = None,
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        func = self.registry.available_functions[tool_name]
        queue: asyncio.Queue[Any] = asyncio.Queue()
        done = object()
        loop = asyncio.get_running_loop()

        def emit(payload: Dict[str, Any]) -> None:
            event = _normalize_tool_ui_event(payload)
            loop.call_soon_threadsafe(queue.put_nowait, event)

        error_holder = {"error": None}
        
        async def run_tool() -> Any:
            try:
                return await self._invoke_tool_callable(func, args, emit)
            except Exception as exc:
                logger.exception(f"Error executing tool {tool_name}")
                error_holder["error"] = exc
                return None
            finally:
                loop.call_soon(queue.put_nowait, done)

        task = asyncio.create_task(run_tool())
        if session is not None:
            session.attach_task(task)

        try:
            while True:
                item = await queue.get()
                if item is done:
                    break
                yield ChatStreamEvent(tool_ui=item)
                if session and session.stop_requested:
                    return

            try:
                result = await task
            except asyncio.CancelledError:
                if session and session.stop_requested:
                    return
                raise

            # Check if there was an error
            if error_holder["error"]:
                # Keep the error format consistent with existing tests
                error_msg = f"Error: {error_holder['error']}"
                yield ChatStreamEvent(
                    tool_call=ToolCallEvent(tool_name=tool_name, result=error_msg)
                )
                return  # Don't yield the normal result event

            yield ChatStreamEvent(
                tool_call=ToolCallEvent(tool_name=tool_name, result=result)
            )
        finally:
            if session is not None:
                session.detach_task(task)

    async def execute_parallel_tool(
        self,
        tool_uses: List[Dict[str, Any]],
        emit: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        async def run_tool(tool_use: Dict[str, Any]) -> Dict[str, Any]:
            recipient_name = tool_use.get("recipient_name")
            if not recipient_name:
                return {"error": "Missing recipient_name", "tool_use": tool_use}

            params = tool_use.get("parameters") or {}
            tool_name = _normalize_tool_name(recipient_name)
            params = self.registry.filter_args(tool_name, params)

            if tool_name == "multi_tool_use.parallel":
                return {
                    "recipient_name": recipient_name,
                    "error": "multi_tool_use.parallel cannot invoke itself",
                }
            if tool_name not in self.registry.available_functions:
                return {
                    "recipient_name": recipient_name,
                    "error": f"Function {tool_name} does not exist",
                }

            try:
                result = await self._invoke_tool_callable(
                    self.registry.available_functions[tool_name],
                    params,
                    emit,
                )
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

    def _register_parallel_tool(self) -> None:
        self.registry.register_builtin(
            name="multi_tool_use.parallel",
            description="Run multiple tool calls in parallel.",
            parameters={
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
            func=self.execute_parallel_tool,
        )

    def _tool_accepts_emit(self, func: Callable[..., Any]) -> bool:
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
