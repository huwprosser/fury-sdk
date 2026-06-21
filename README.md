<p align="center">
  <img src="https://raw.githubusercontent.com/huwprosser/fury/a5f785da526e09af78d9522f1b275be421bbb5e8/fury.png" alt="Fury Logo" width="192" />
</p>

<h1 align="center">Fury</h1>

<p align="center">
  <a href="https://discord.gg/xC9Yd6VH2a">
    <img src="https://img.shields.io/discord/841085263266447400?logo=discord" alt="Discord">
  </a>
  <a href="https://github.com/huwprosser/fury/actions/workflows/tests.yml">
    <img src="https://github.com/huwprosser/fury/actions/workflows/tests.yml/badge.svg?branch=main" alt="Tests">
  </a>
</p>

A flexible and powerful AI agent library for Python, designed to build agents with tool support, multimodal capabilities, and streaming responses.

## Features
- **Interruption and early stopping**: Agents now use the Runner pattern, allowing them to be interrupted or stopped mid-generation.
- **Tool Support**: Define and register custom tools (functions) that the agent can execute and parallel tool execution support.
- **Image inputs**: Support for multimodal image inputs.
- **History Management**: Use `HistoryManager` for simple target-context trimming.


## Installation

Install with uv:

```bash
uv add fury-sdk
```

## Quick Start

```python
from fury import Agent

agent = Agent(
    model="unsloth/GLM-4.6V-Flash-GGUF:Q8_0",
    system_prompt="You are a helpful assistant.",
)

print(agent.ask("Hello!", history=[]))
print(agent.ask("Hello!", history=[], model="another-model"))
```

Other examples:
- [Basic Chat Loop](examples/chat.py)
- [Markdown Memory Tool](examples/memory_chat.py)
- [History Compaction](examples/history_compaction.py)
- [Interruption](examples/interruption.py)
- [Coding Assistant](examples/coding-assistant)

## History Management
Fury's `HistoryManager` is intentionally small: it stores a list of OpenAI-compatible JSON message dictionaries and trims old messages to keep the newest context within a target size.

```python
from fury import HistoryManager

history_manager = HistoryManager(target_context_length=4096)

await history_manager.add({"role": "user", "content": user_input})

transcript = []
runner = agent.runner()
async for event in runner.chat(history_manager.history):
    if event.history_delta:
        transcript.append(event.history_delta.message)
await history_manager.extend(transcript)
```

Image messages are supported via `add_image()`. By default Fury stores a lightweight placeholder plus path metadata; set `save_images_to_history=True` to keep the full image payload.

See [docs/history_manager.md](docs/history_manager.md).

### Configuration Options

```python
agent = Agent(
    model="your-model-name",
    system_prompt="You are a helpful assistant.",
    parallel_tool_calls=False,
    generation_params={
        "temperature": 0.2,
        "max_tokens": 512,
    },
)

# Disable reasoning stream content (default is False)
runner = agent.runner()
async for event in runner.chat(history, reasoning=False):
    ...

```

### Persisting Transcripts

Fury separates UI stream events from model-visible transcript events. Persist `event.history_delta.message` to save the exact OpenAI-compatible messages Fury used for tool calls, tool results, multimodal follow-ups, and final assistant replies.

```python
transcript = []

async for event in agent.runner().chat(history):
    if event.content:
        print(event.content, end="")
    if event.history_delta:
        transcript.append(event.history_delta.message)

save_messages(transcript)
```

For non-streaming collection:

```python
result = await agent.runner().complete(history)
save_messages(result.transcript)
print(result.content)
```

### Defining Tools

You can give your agent tools to interact with the world. Tools are `Tool` objects.

Input and output schemas help the model to correctly pass parameters through to the function. Fury will automatically prune any hallucinated parameters not defined in the input schema.

Learn more in the [OpenAI guide](https://developers.openai.com/api/docs/guides/function-calling/)

```python
from fury import Agent, Tool


def add(a: int, b: int):
    return {"result": a + b}

# Create the tool
add_tool = Tool(
    name="add",
    description="Add two numbers together",
    execute=add,
    input_schema={
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    },
    output_schema={
        "type": "object",
        "properties": {"result": {"type": "integer"}},
        "required": ["result"],
    },
)

# Pass to agent
agent = Agent(..., tools=[add_tool])
```

If your tool accepts an `emit` parameter, Fury injects a runtime-only callback during execution so the tool can stream structured UI events during tool execution.

```python
def search(query: str, emit):
    emit({"id": "search-1", "title": f"Searching for {query}", "type": "tool_call"})
    return {"query": query}
```

These arrive in the chat stream as `event.tool_ui`, separate from `event.tool_call`.

### Coding Assistant Example

Check out [examples/coding-assistant/coding_assistant.py](examples/coding-assistant/coding_assistant.py) for a full-featured example that includes:

- **Tools**: File system operations (`read`, `write`, `edit`, `bash`).
- **Skills System**: Loading specialized capabilities from `SKILL.md` files.
- **Markdown Memory Example**: A simple editable markdown file injected into the system prompt.
- **History Manager**: Uses `HistoryManager` to summarize long conversations and save context window.

Build something neat.
