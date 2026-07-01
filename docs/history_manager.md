# HistoryManager

`HistoryManager` is a small bounded list for OpenAI-compatible chat messages.

It accepts message dictionaries such as `{ "role": "user", "content": "hello" }`, including multimodal/image message content, and keeps only the newest messages that fit inside `target_context_length`.

No summarisation, persistence, IDs, variants, or regeneration are built in.

```python
from fury import HistoryManager

history = HistoryManager(target_context_length=4096)
await history.add({"role": "user", "content": "Hello"})
await history.extend([
    {"role": "assistant", "content": "Hi!"},
])

runner = agent.runner()
async for event in runner.chat(history.history):
    ...
```

## Images

```python
await history.add_image("./image.png", text="What is this?")
```

By default image history stores a lightweight placeholder plus path metadata. Use `save_images_to_history=True` to keep the full image payload.
