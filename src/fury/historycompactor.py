from __future__ import annotations

from typing import Any, Dict, List, Optional

from .agent import Agent
from .utils.validation import validate_history

DEFAULT_COMPACTION_PROMPT = """Summarize this chat for future turns.
Keep the summary concise but preserve:
- the user's goals and preferences
- important decisions and constraints
- facts that would be needed to continue the conversation
- projects, plans, promises, and unresolved questions
- relevant files, URLs, or tool results

Do not include filler. Do not mention that this is an automatic compaction."""


class HistoryCompactor:
    """Generate a compact string summary for a list of chat messages."""

    def __init__(
        self,
        agent: Agent,
        *,
        prompt: str = DEFAULT_COMPACTION_PROMPT,
    ) -> None:
        self.agent = agent
        self.prompt = prompt

    async def compact(
        self,
        history: List[Dict[str, Any]],
        *,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Summarize ``history`` into a single string."""
        validate_history(history)
        compaction_prompt = prompt if prompt is not None else self.prompt
        messages = [
            *history,
            {"role": "user", "content": compaction_prompt},
        ]
        result = await self.agent.runner().complete(messages, reasoning=False, model=model)
        return result.content.strip()

    async def compaction_summary(
        self,
        history: List[Dict[str, Any]],
        *,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Alias for ``compact()``."""
        return await self.compact(history, prompt=prompt, model=model)
