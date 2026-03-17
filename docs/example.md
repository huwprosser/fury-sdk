# StaticHistoryManager Example

Use `StaticHistoryManager` when you want a strict token budget with no summarization.
It keeps only the newest messages that fit within `target_context_length` and drops older
ones automatically.

```python
import asyncio
from fury import Agent, StaticHistoryManager

agent = Agent(
    model="your-model-name",
    system_prompt="You are a helpful assistant.",
)

history_manager = StaticHistoryManager(
    target_context_length=2000,
    history=[{"role": "system", "content": "You are concise."}],
)


async def main() -> None:
    await history_manager.add({"role": "user", "content": "Summarize this long text..."})

    buffer = ""
    runner = agent.runner()
    async for event in runner.chat(history_manager.history, reasoning=False):
        if event.content:
            buffer += event.content
            print(event.content, end="", flush=True)

    await history_manager.add({"role": "assistant", "content": buffer})


if __name__ == "__main__":
    asyncio.run(main())
```

With this manager, history is always kept at or below the configured target context length.
