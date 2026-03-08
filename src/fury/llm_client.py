from __future__ import annotations

from typing import Any, Literal, Optional, Protocol, runtime_checkable

from .openai_httpx import AsyncOpenAI as HttpxAsyncOpenAI


@runtime_checkable
class ChatCompletionsProtocol(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


@runtime_checkable
class ChatNamespaceProtocol(Protocol):
    completions: ChatCompletionsProtocol


@runtime_checkable
class ChatClientProtocol(Protocol):
    chat: ChatNamespaceProtocol

    async def close(self) -> None: ...


ClientBackend = Literal["openai", "httpx"]


def create_chat_client(
    *,
    base_url: str,
    api_key: str = "",
    backend: ClientBackend = "openai",
    **kwargs: Any,
) -> ChatClientProtocol:
    if backend == "httpx":
        return HttpxAsyncOpenAI(base_url=base_url, api_key=api_key, **kwargs)

    try:
        from openai import AsyncOpenAI as SdkAsyncOpenAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The OpenAI SDK is not installed. Use client_backend='httpx' or "
            "install the 'openai' dependency."
        ) from exc

    return SdkAsyncOpenAI(base_url=base_url, api_key=api_key, **kwargs)
