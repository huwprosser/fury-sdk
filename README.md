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
- **[New] Durable Memory**: Persist named memory scopes, bind them to individual agents, and expose an airgapped memory tool.
- **Interruption and early stopping**: Agents now use the Runner pattern, allowing them to be interrupted or stopped mid-generation.
- **Tool Support**: Define and register custom tools (functions) that the agent can execute and parallel tool execution support.
- **Image and Voice inputs**: Support for image and voice inputs (using Whisper for STT).
- **Text-to-Speech (TTS)**: Generate audio with NeuTTS via `Agent.speak()`.
- **History Management**: Use `HistoryManager` for auto-compaction support or `StaticHistoryManager` for strict fixed-size context trimming.


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
- [Chat With Durable Memory](examples/memory_chat.py)
- [Interruption](examples/interruption.py)
- [Coding Assistant](examples/coding-assistant)
- [Voice Chat](examples/voice_chat.py)
- [Text-to-speech](examples/tts.py)

## History Management
Fury makes managing history limits easy by providing simple, built-in history managers. They are just list managers that monitor context utilization and trim or compact your list accordingly.

The standard `HistoryManager` will auto-compact your history as you add messages to it (summarise using an Agent) in a similar way to Claude Code, Codex and Pi.

```python
from fury import HistoryManager

history_manager = HistoryManager(agent=agent)

# Add something to history like this:
await history_manager.add({"role": "user", "content": user_input})

# Use the history like this:
runner = agent.runner()
async for event in runner.chat(history_manager.history):
    # ...
```

See [examples/chat.py](examples/chat.py) for a full working example.

If you do not want auto-compaction and a hard history limit, use `StaticHistoryManager`:

```python
from fury import StaticHistoryManager

history_manager = StaticHistoryManager(
    target_context_length=4096,
    history=[{"role": "system", "content": "You are helpful."}],
)
```

It keeps only the newest messages that fit in the target context length.
See [docs/example.md](docs/example.md) for a complete example.

## Durable Memory

Fury can persist durable memory outside the current chat history using explicit named scopes. Pass `memory_scope` when constructing an agent and Fury will:

- create or reuse a `MemoryStore`
- inject the latest memory snapshot for that scope into the system prompt
- register a `memory` tool bound only to that scope

```python
from fury import Agent

agent = Agent(
    model="your-model-name",
    system_prompt="You are a helpful assistant.",
    memory_scope="my-project",
)
```

To share one backing store across multiple airgapped agents:

```python
from fury import Agent, MemoryStore

store = MemoryStore(".fury/memory")

alpha = Agent(
    model="your-model-name",
    system_prompt="You help project alpha.",
    memory_store=store,
    memory_scope="project-alpha",
)

beta = Agent(
    model="your-model-name",
    system_prompt="You help project beta.",
    memory_store=store,
    memory_scope="project-beta",
)
```

See [docs/memory.md](docs/memory.md) for the full API.

### Configuration Options

```python
agent = Agent(
    model="your-model-name",
    system_prompt="You are a helpful assistant.",
    parallel_tool_calls=False,
    disable_stt=False,
    disable_tts=False,
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

### Defining Tools

You can give your agent tools to interact with the world. Tools are defined using the `create_tool` helper.

Input and output schemas help the model to correctly pass parameters through to the function. Fury will automatically prune any hallucinated parameters not defined in the input schema.

Learn more in the [OpenAI guide](https://developers.openai.com/api/docs/guides/function-calling/)

```python
from fury import Agent, create_tool


def add(a: int, b: int):
    return {"result": a + b}

# Create the tool
add_tool = create_tool(
    id="add",
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
- **Memory System**: Using durable memory plus `SOUL.md` for project context.
- **History Manager**: Uses `HistoryManager` to summarize long conversations and save context window.

Build something neat.
