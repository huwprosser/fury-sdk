from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional
from urllib.parse import urljoin

import httpx


@dataclass
class ChatCompletionChunk:
    choices: List[Any]


class APIStatusError(Exception):
    """Raised when the API returns a non-2xx response."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.status_code = response.status_code
        self.body = response.text
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        body = self.body.strip()
        if not body:
            return (
                f"OpenAI-compatible API request failed with status {self.status_code}"
            )
        return (
            f"OpenAI-compatible API request failed with status {self.status_code}: "
            f"{body}"
        )


class AsyncStreamChatCompletions(AsyncIterator[ChatCompletionChunk]):
    """
    Async iterator for SSE chat completion streams.

    This intentionally implements only the subset of the wire format that Fury
    currently reads:
    - choices[0].delta.content
    - choices[0].delta.reasoning_content
    - choices[0].delta.tool_calls
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        retry_config: "RetryConfig",
    ) -> None:
        self._client = client
        self._url = url
        self._payload = payload
        self._headers = headers
        self._retry_config = retry_config
        self._response: Optional[httpx.Response] = None
        self._lines: Optional[AsyncIterator[str]] = None
        self._attempt = 0
        self._has_yielded = False

    async def __anext__(self) -> ChatCompletionChunk:
        while True:
            if self._response is None:
                await self._open_stream()
                if self._response is None:
                    continue

            if self._lines is None:
                await self.aclose()
                raise RuntimeError(
                    "Fury opened a chat stream without initializing the SSE iterator."
                )

            try:
                async for line in self._lines:
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if data == "[DONE]":
                        await self.aclose()
                        raise StopAsyncIteration

                    payload = json.loads(data)
                    chunk = _parse_stream_chunk(payload)
                    if chunk is None:
                        continue
                    self._has_yielded = True
                    return chunk
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ):
                if self._has_yielded:
                    await self.aclose()
                    raise
                if not await self._retry_stream_connection():
                    raise

            await self.aclose()
            raise StopAsyncIteration

    async def aclose(self) -> None:
        if self._response is not None:
            await self._response.aclose()
            self._response = None
            self._lines = None

    async def _open_stream(self) -> None:
        request = self._client.build_request(
            "POST",
            self._url,
            headers=self._headers,
            json=self._payload,
        )
        try:
            self._response = await self._client.send(request, stream=True)
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
            if not await self._retry_stream_connection():
                raise
            return

        if self._response.is_error:
            if _should_retry_response(
                self._response
            ) and await self._retry_stream_response(self._response):
                return
            body = await self._response.aread()
            self._response._content = body
            await self._response.aclose()
            raise APIStatusError(self._response)

        self._lines = self._response.aiter_lines()

    async def _retry_stream_connection(self) -> bool:
        delay = _compute_retry_delay(
            attempt=self._attempt,
            retry_config=self._retry_config,
            response=None,
        )
        if delay is None:
            await self.aclose()
            return False
        self._attempt += 1
        await self.aclose()
        await self._sleep(delay)
        return True

    async def _retry_stream_response(self, response: httpx.Response) -> bool:
        delay = _compute_retry_delay(
            attempt=self._attempt,
            retry_config=self._retry_config,
            response=response,
        )
        if delay is None:
            return False
        self._attempt += 1
        await response.aclose()
        self._response = None
        self._lines = None
        await self._sleep(delay)
        return True

    async def _sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class AsyncChatCompletions:
    """Minimal chat completions resource."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        retry_config: "RetryConfig",
    ) -> None:
        self._client = client
        self._url = _join_url(base_url, "chat/completions")
        self._headers = _build_headers(api_key)
        self._retry_config = retry_config

    async def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        payload: Dict[str, Any] = {"model": model, "messages": messages}
        if tools is not None:
            payload["tools"] = tools
        if extra_body:
            payload["extra_body"] = extra_body
        payload.update(kwargs)

        if stream:
            payload["stream"] = True
            return AsyncStreamChatCompletions(
                client=self._client,
                url=self._url,
                payload=payload,
                headers=self._headers,
                retry_config=self._retry_config,
            )
        response = await _request_with_retries(
            client=self._client,
            method="POST",
            url=self._url,
            headers=self._headers,
            json_body=payload,
            retry_config=self._retry_config,
        )
        return _parse_chat_completion_response(response.json())


class AsyncChat:
    """Namespace wrapper matching client.chat.completions."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        retry_config: "RetryConfig",
    ) -> None:
        self.completions = AsyncChatCompletions(client, base_url, api_key, retry_config)


@dataclass
class RetryConfig:
    max_retries: int = 3
    initial_delay: float = 0.5
    max_delay: float = 8.0
    backoff_factor: float = 2.0
    jitter: float = 0.25


class AsyncOpenAICompatibleClient:
    """
    Small httpx-backed client for the subset of the OpenAI-compatible API Fury uses.

    Supported surface:
    - AsyncOpenAICompatibleClient(base_url=..., api_key=...)
    - client.chat.completions.create(...)
    - non-streaming text responses
    - streaming text/reasoning/tool-call deltas over SSE

    Intentionally omitted:
    - retries/backoff
    - websocket/realtime APIs
    - typed request validation
    - pagination
    - all endpoints besides chat completions
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8080/v1",
        api_key: str = "",
        timeout: float = 60.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        max_retries: int = 3,
        initial_retry_delay: float = 0.5,
        max_retry_delay: float = 8.0,
        retry_backoff_factor: float = 2.0,
        retry_jitter: float = 0.25,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._owns_client = http_client is None
        self._retry_config = RetryConfig(
            max_retries=max_retries,
            initial_delay=initial_retry_delay,
            max_delay=max_retry_delay,
            backoff_factor=retry_backoff_factor,
            jitter=retry_jitter,
        )
        self._client = http_client or httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            ),
        )
        self.chat = AsyncChat(
            self._client,
            base_url,
            api_key,
            self._retry_config,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncOpenAICompatibleClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()


def _build_headers(api_key: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _join_url(base_url: str, path: str) -> str:
    normalized = base_url.rstrip("/") + "/"
    return urljoin(normalized, path)


def _ns(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def _parse_chat_completion_response(payload: Dict[str, Any]) -> Any:
    raw_choices = payload.get("choices") or [{}]
    choices = []

    for raw_choice in raw_choices:
        message = raw_choice.get("message") or {}
        choices.append(_ns(message=_ns(content=message.get("content"))))

    return _ns(choices=choices)


def _parse_stream_chunk(payload: Dict[str, Any]) -> Optional[ChatCompletionChunk]:
    raw_choices = payload.get("choices") or []
    if not raw_choices:
        return None

    parsed_choices: List[Any] = []
    for raw_choice in raw_choices:
        raw_delta = raw_choice.get("delta") or {}
        parsed_choices.append(
            _ns(
                delta=_ns(
                    content=raw_delta.get("content"),
                    reasoning_content=raw_delta.get("reasoning_content"),
                    tool_calls=_parse_tool_call_deltas(raw_delta.get("tool_calls")),
                )
            )
        )

    return ChatCompletionChunk(choices=parsed_choices)


def _parse_tool_call_deltas(raw_tool_calls: Any) -> Optional[List[Any]]:
    if not isinstance(raw_tool_calls, list):
        return None

    deltas: List[Any] = []
    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, dict):
            continue
        function = raw_tool_call.get("function") or {}
        deltas.append(
            _ns(
                index=int(raw_tool_call.get("index", 0)),
                id=raw_tool_call.get("id"),
                function=_ns(
                    name=function.get("name"),
                    arguments=function.get("arguments"),
                ),
            )
        )

    return deltas or None


async def _request_with_retries(
    *,
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: Dict[str, str],
    json_body: Dict[str, Any],
    retry_config: RetryConfig,
) -> httpx.Response:
    attempt = 0
    while True:
        try:
            response = await client.request(
                method,
                url,
                headers=headers,
                json=json_body,
            )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
            delay = _compute_retry_delay(
                attempt=attempt,
                retry_config=retry_config,
                response=None,
            )
            if delay is None:
                raise
            attempt += 1
            await _async_sleep(delay)
            continue

        if not response.is_error:
            return response

        delay = _compute_retry_delay(
            attempt=attempt,
            retry_config=retry_config,
            response=response,
        )
        if delay is None:
            raise APIStatusError(response)
        attempt += 1
        await response.aclose()
        await _async_sleep(delay)


def _should_retry_response(response: httpx.Response) -> bool:
    return response.status_code in {429, 500, 502, 503, 504}


def _compute_retry_delay(
    *,
    attempt: int,
    retry_config: RetryConfig,
    response: Optional[httpx.Response],
) -> Optional[float]:
    if attempt >= retry_config.max_retries:
        return None

    if response is not None and not _should_retry_response(response):
        return None

    retry_after = _parse_retry_after(response)
    if retry_after is not None:
        return min(retry_after, retry_config.max_delay)

    base_delay = min(
        retry_config.initial_delay * (retry_config.backoff_factor**attempt),
        retry_config.max_delay,
    )
    jitter = random.uniform(0.0, retry_config.jitter)
    return min(base_delay + jitter, retry_config.max_delay)


def _parse_retry_after(response: Optional[httpx.Response]) -> Optional[float]:
    if response is None:
        return None

    raw_value = response.headers.get("Retry-After")
    if not raw_value:
        return None

    try:
        return max(0.0, float(raw_value))
    except ValueError:
        pass

    try:
        retry_time = parsedate_to_datetime(raw_value)
    except (TypeError, ValueError):
        return None

    delay = retry_time.timestamp() - time.time()
    return max(0.0, delay)


async def _async_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


AsyncOpenAI = AsyncOpenAICompatibleClient
