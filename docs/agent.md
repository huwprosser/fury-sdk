# Agent

The `Agent` class is the core runtime for chatting with an OpenAI-compatible model while supporting tool calling, multimodal inputs, and streaming responses. It provides single-shot (`ask`, `ask_async`) interfaces and creates `Runner` objects for streaming runs.

## Key Features

- **Streaming chat** via `Runner.chat()`.
- **Interruptible generation** via `Runner.cancel()` and `Runner.interrupt()`.
- **Tool calling** through `create_tool()` and registered tools.
- **Parallel tool execution** using the built-in `multi_tool_use.parallel` wrapper.
- **Multimodal inputs** with helper methods for images and voice messages.
- **Optional TTS** via `Agent.speak()`.

## Basic Usage

```python
from fury import Agent

agent = Agent(
    model="your-model-name",
    system_prompt="You are a helpful assistant.",
    base_url="http://127.0.0.1:8080/v1",
    api_key="your-api-key",
)

response = agent.ask("Hello!", history=[])
print(response)
```

## Streaming Chat

```python
import asyncio
from fury import Agent

agent = Agent(
    model="your-model-name",
    system_prompt="You are a helpful assistant.",
)

async def main():
    history = [{"role": "user", "content": "Hello"}]
    runner = agent.runner()
    async for event in runner.chat(history, reasoning=False):
        if event.content:
            print(event.content, end="", flush=True)

asyncio.run(main())
```

## Cancelling Or Interrupting A Generation

Use `agent.runner()` when you want to stream a reply and optionally stop it before completion.

- `runner.cancel()`: stops the in-flight request and discards the partial assistant response from history.
- `runner.interrupt()`: stops the in-flight request and preserves the partial assistant response by appending it to the provided history.
- For the minimal version, see `docs/interruption.md`.
- For a runnable example, see `examples/interruption.py`.

```python
import asyncio
from fury import Agent

agent = Agent(
    model="your-model-name",
    system_prompt="You are a helpful assistant.",
)

async def main():
    history = [{"role": "user", "content": "Explain TCP in detail."}]
    runner = agent.runner()

    async for event in runner.chat(history):
        if event.content:
            print(event.content, end="", flush=True)
            runner.interrupt()

    print("\nPartial response:", runner.partial_response)
    print("Updated history:", history)

asyncio.run(main())
```

## Tool Calling

Tools are defined with `create_tool()` and passed to the agent. The agent filters any extra arguments that the model hallucinates to match the declared input schema.

```python
from fury import Agent, create_tool

def add(a: int, b: int, emit=None):
    if emit:
        emit(
            {
                "id": "add",
                "title": f"Adding {a} and {b}",
                "type": "tool_call",
            }
        )
    return {"result": a + b}

add_tool = create_tool(
    id="add",
    description="Add two numbers together",
    execute=add,
    input_schema={
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
    output_schema={
        "type": "object",
        "properties": {"result": {"type": "integer"}},
        "required": ["result"],
    },
)

agent = Agent(
    model="your-model-name",
    system_prompt="You are a helpful assistant.",
    tools=[add_tool],
)
```

If a tool function accepts an `emit` argument, Fury injects a runtime-only callback while the tool runs. `emit` is not added to the tool schema sent to the model.

```python
def search(query: str, emit):
    emit(
        {
            "id": "search-1",
            "title": f"Searching for {query}",
            "type": "tool_call",
        }
    )
    return {"query": query}
```

Stream consumers receive these as `ChatStreamEvent(tool_ui=...)`, separate from `tool_call` arguments and results.

## History Management

`Agent` does not automatically manage history. Pass a list of `{role, content}` messages into `chat()` or `ask()`. For auto-compaction, use `HistoryManager` (see `docs/history_manager.md`).

## Multimodal Helpers

For managed histories, prefer:

- `await history_manager.add_image(image_path, text="...")`
- `await history_manager.add_voice(base64_audio_bytes)`

`Agent` still exposes lower-level helpers for direct list-based history management.

## Text-to-Speech

`Agent.speak()` uses the NeuTTS backend to generate audio from text, conditioned on a reference clip. See `examples/tts.py` for a complete example.

## Constructor Arguments

- `model`: Model name.
- `system_prompt`: System instruction string.
- `tools`: List of `Tool` objects from `create_tool()`.
- `base_url`: OpenAI-compatible server URL.
- `api_key`: API key for the server.
- `generation_params`: Additional model parameters (temperature, max_tokens, etc.).
- `max_tool_rounds`: Maximum tool-call iterations per request.
- `parallel_tool_calls`: Enable the built-in parallel tool wrapper.
- `tts_provider`: Optional custom TTS provider.
