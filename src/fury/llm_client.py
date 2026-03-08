from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .openai_httpx import AsyncOpenAICompatibleClient


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

def create_chat_client(
    *,
    base_url: str,
    api_key: str = "",
    **kwargs: Any,
) -> ChatClientProtocol:
    return AsyncOpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        **kwargs,
    )
