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

To persist a session transcript as JSONL and reload it on startup:

```python
history_manager = HistoryManager(
    agent,
    persist_to_disk=True,
    session_id="demo-session",
)
```

This writes raw messages to `.fury/history/*.jsonl`. On initialization, if the
session file already exists, `HistoryManager` loads it back into `history`.
If `auto_compact=True`, the reloaded working set is compacted in memory on the
first async `add()` or `extend()` call.
When compaction runs, `HistoryManager` prints a short status line by default.

## Message IDs and Variants

`HistoryManager` assigns every managed message a stable `id`. The ID is stored
with persisted JSONL history and is preserved across reloads, so callers can edit,
delete, regenerate, or select variants without relying on list indexes.

```python
await history_manager.add({"role": "user", "content": "Hello"})

message_id = history_manager.history[-1]["id"]
history_manager.edit_message(
    message_id,
    {"role": "user", "content": "Hello, rewritten"},
)
history_manager.delete_message(message_id)
```

Internal history metadata such as `id`, `variants`, and `active_variant_id` is
removed before messages are sent to the model. Your application can safely use
those fields in `history_manager.history` without leaking them into provider
requests.

### Regenerating a Message

`regenerate_message()` reruns `Agent.chat()` for an existing assistant message,
using the history before that message as context. The regenerated response is
stored as a new variant, made active, and copied back onto the top-level message
so `history_manager.history` remains a normal linear chat transcript.

```python
await history_manager.add({"role": "user", "content": "Explain queues."})
await history_manager.add({"role": "assistant", "content": "Original answer."})

assistant_id = history_manager.history[-1]["id"]

await history_manager.regenerate_message(assistant_id)

message = history_manager.history[-1]
print(message["content"])          # active regenerated answer
print(message["variants"])         # original and regenerated versions
print(message["active_variant_id"])
```

`regenerate_message()` requires `HistoryManager(agent=agent, ...)` because it uses
the bound agent to generate the new response. It currently targets assistant
messages only.

### Switching Variants

Use `set_variant()` to switch the active version of a message:

```python
assistant_id = history_manager.history[-1]["id"]
first_variant_id = history_manager.history[-1]["variants"][0]["id"]

history_manager.set_variant(assistant_id, first_variant_id)
```

The active variant's message fields are copied to the top-level message while the
same message `id` and full `variants` list are preserved. This makes it cheap for
UIs to let users move between generated alternatives without changing downstream
code that reads the linear history.

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
- `persist_to_disk`: Append raw messages to a JSONL transcript on disk.
- `session_id`: Session identifier used to name and reload the JSONL file.
- `history_root`: Directory used for persisted history files (default: `.fury/history`).
- `show_compaction_status`: Print a short compaction notice (defaults to `False` when the bound agent uses `suppress_logs=True`, otherwise `True`).
- `save_images_to_history`: Persist full base64 image payloads in managed history. Defaults to `False`, which stores a lightweight `[The user shared an image]` placeholder plus image-path metadata and materializes the real image only when building model input.

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

`edit_message()` and `delete_message()` are preferred over mutating
`history_manager.history` directly when persistence is enabled, because they
rewrite the JSONL file atomically after changing the managed list.
