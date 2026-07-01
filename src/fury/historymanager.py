from __future__ import annotations

from typing import Any, Dict, List, Optional

from .multimodal import build_image_history_message
from .utils.validation import validate_message


def estimate_message_tokens(message: Dict[str, Any]) -> int:
    """Very small token estimate: roughly four characters per token."""
    content = message.get("content", "")
    if isinstance(content, str):
        chars = len(content)
    else:
        chars = len(str(content))
    return max(1, chars // 4)


class HistoryManager:
    """Simple bounded chat history manager.

    Stores OpenAI-compatible message dictionaries and keeps only the newest
    messages that fit within ``target_context_length``. Messages may contain
    plain text or multimodal/image content.
    """

    def __init__(
        self,
        history: Optional[List[Dict[str, Any]]] = None,
        *,
        target_context_length: int = 32768,
        save_images_to_history: bool = False,
        **_: Any,
    ) -> None:
        if target_context_length <= 0:
            raise ValueError("target_context_length must be greater than zero")

        self.target_context_length = target_context_length
        self.save_images_to_history = save_images_to_history
        self.history: List[Dict[str, Any]] = []
        self.set(history or [])

    async def add(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Append one message and trim to the target context length."""
        self.history.append(self._prepare_message(message))
        self.reduce()
        return self.history

    async def extend(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Append messages and trim to the target context length."""
        self.history.extend(self._prepare_message(message) for message in messages)
        self.reduce()
        return self.history

    async def add_image(
        self,
        image_path: str,
        *,
        text: str = "Image input.",
    ) -> List[Dict[str, Any]]:
        """Append an image message and trim to the target context length."""
        return await self.add(
            build_image_history_message(
                image_path,
                text=text,
                save_image=self.save_images_to_history,
            )
        )

    def set(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Replace the managed history and trim it."""
        self.history = [self._prepare_message(message) for message in history]
        self.reduce()
        return self.history

    def clear(self) -> List[Dict[str, Any]]:
        """Remove all messages."""
        self.history.clear()
        return self.history

    def reduce(self) -> List[Dict[str, Any]]:
        """Trim history in-place to the newest messages fitting the target."""
        self.history = self._fit_to_target(self.history)
        return self.history

    def get_context_usage(self) -> tuple[int, float]:
        tokens = sum(self._estimate_tokens_for_message(msg) for msg in self.history)
        percent = (
            (tokens / self.target_context_length) * 100
            if self.target_context_length
            else 0.0
        )
        return tokens, percent

    def _prepare_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        prepared = dict(message)
        validate_message(prepared)
        return prepared

    def _fit_to_target(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        fitted: List[Dict[str, Any]] = []
        total_tokens = 0

        for message in reversed(messages):
            message_tokens = self._estimate_tokens_for_message(message)
            if fitted and total_tokens + message_tokens > self.target_context_length:
                break
            if not fitted and message_tokens > self.target_context_length:
                fitted = [message]
                break
            fitted.append(message)
            total_tokens += message_tokens

        return list(reversed(fitted))

    def _estimate_tokens_for_message(self, message: Dict[str, Any]) -> int:
        return estimate_message_tokens(message)
