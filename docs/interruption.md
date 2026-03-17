# Interruption

Use a `Runner` when you want to stop a reply that is still streaming.

- `runner.cancel()` stops the reply and discards the partial assistant text from `history`.
- `runner.interrupt()` stops the reply and keeps the partial assistant text in `history`.

Minimal example:

```python
import asyncio
from fury import Agent

agent = Agent(
    model="your-model-name",
    system_prompt="You are a helpful assistant.",
)


async def main() -> None:
    history = [{"role": "user", "content": "Explain TCP in detail."}]
    runner = agent.runner()

    async for event in runner.chat(history):
        if event.content:
            print(event.content, end="", flush=True)
            runner.interrupt()

    print("\nPartial response:", runner.partial_response)
    print("History:", history)


if __name__ == "__main__":
    asyncio.run(main())
```

After `interrupt()`, Fury appends the partial assistant response to the same `history` list.

For a runnable chat-loop example with a hotkey, see `examples/interruption.py`.
