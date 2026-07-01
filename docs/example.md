# HistoryManager Example

Use `HistoryManager` when you want a strict target context length.

```python
import asyncio
from fury import Agent, HistoryManager

agent = Agent(
    model="your-model-name",
    system_prompt="You are helpful.",
)

history_manager = HistoryManager(
    target_context_length=4096,
    history=[{"role": "system", "content": "You are helpful."}],
)

async def main():
    await history_manager.add({"role": "user", "content": "Hello"})

    transcript = []
    async for event in agent.runner().chat(history_manager.history):
        if event.content:
            print(event.content, end="", flush=True)
        if event.history_delta:
            transcript.append(event.history_delta.message)

    await history_manager.extend(transcript)

asyncio.run(main())
```
