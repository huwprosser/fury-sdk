# History Manager

The `HistoryManager` manages conversation history (a list of `{role, content}` messages) and can automatically compact older messages into a summary when the context window gets tight. It is designed to drop into the same history list used by `Runner.chat()` and keeps the tail of recent messages intact.

For strict non-summarizing history limits, use `StaticHistoryManager`. It keeps only the newest messages that fit inside a fixed `target_context_length`.

## How It Works

`HistoryManager` estimates token usage by counting characters and dividing by four (roughly 4 chars per token). When the estimated total exceeds the configured context window minus the reserved tokens, it:

1. Identifies a cut index so that the most recent messages (based on `keep_recent_tokens`) are preserved.
2. Summarizes the earlier portion via the same OpenAI-compatible Fury transport client used by `Agent`.
3. Replaces the summarized portion with a system message containing the summary.

The summary message uses a configurable `summary_prefix` so it can be recognized and updated on subsequent compactions.

## Usage

```python
import asyncio
from fury import Agent, HistoryManager

agent = Agent(
    model="your-model-name",
    system_prompt="You are a helpful assistant.",
)

history_manager = HistoryManager(
    agent=agent,
    auto_compact=True,
    context_window=32768,
    reserve_tokens=8192,
    keep_recent_tokens=8000,
)

async def main():
    await history_manager.add({"role": "user", "content": "Hello"})
    runner = agent.runner()
    async for event in runner.chat(history_manager.history, reasoning=False):
        if event.content:
            print(event.content, end="", flush=True)
    await history_manager.add({"role": "assistant", "content": "Hi!"})

asyncio.run(main())
```

For a runnable example, see `examples/chat.py`.

## StaticHistoryManager

`StaticHistoryManager` provides a fixed-size history window with no auto compaction and no summary calls. On initialization and every add/extend operation, it drops older messages until the newest messages fit inside `target_context_length`.

```python
from fury import StaticHistoryManager

history_manager = StaticHistoryManager(
    target_context_length=4096,
    history=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ],
)
```

See `docs/example.md` for a full example.

## Configuration

- `history`: Optional initial list of messages.
- `agent`: Pass an `Agent` to reuse its transport client and model for summaries.
- `client`: Provide an `AsyncOpenAICompatibleClient` directly (required if no `agent`).
- `summary_model`: Model to use for summaries (required if no `agent`).
- `auto_compact`: Toggle auto-compaction on message adds (default: `True`).
- `context_window`: Estimated total token capacity before compaction (default: `32768`).
- `reserve_tokens`: Tokens to keep in reserve for model replies (default: `8192`).
- `keep_recent_tokens`: Tokens to preserve at the tail (default: `8000`).
- `summary_prefix`: Prefix used to store and recognize summary messages.
- `summary_system_prompt`: System prompt for the summary model.

## Manual Control

If you want to manage history manually without compaction, use `add_nowait`:

```python
history_manager.add_nowait({"role": "user", "content": "Hello"})
```

To batch append multiple messages at once (with optional compaction), use `extend`:

```python
await history_manager.extend([
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
])
```
