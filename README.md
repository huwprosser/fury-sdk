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
```

Other examples:
- [Basic Chat Loop](examples/chat.py)
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
async for event in agent.chat(history_manager.history):
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
async for event in agent.chat(history, reasoning=False):
    ...

```

## Advanced Usage

### Text-to-Speech (Based on NeuTTS-Air)

NeuTTS-Air is one of the easiest Autoregressive TTS models to work with right now imo. You may chose not to use this which is why TTS support is an optional additional dependency list. The `neutts_minimal.py` implements a lightweight inference-only TTS engine. It currently depends on eSpeak and llama_cpp to spin up the model locally. PRs are welcome on slimming this down.

Use `Agent.speak()` with a reference audio clip and matching text. The default
backbone and codec are `neuphonic/neutts-air-q4-gguf` and `neuphonic/neucodec-onnx-decoder`.

```python
import numpy as np
import wave
from fury import Agent

agent = Agent(
    model="your-model-name",
    system_prompt="You are a helpful assistant.",
    base_url="http://127.0.0.1:8080/v1",
    api_key="your-api-key",
)

chunks = agent.speak(
    text="Hello from Fury!",
    ref_text="Welcome home sir.",
    ref_audio_path="./examples/resources/ref.wav",
)

audio = np.concatenate(list(chunks))
with wave.open("output.wav", "wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(24000)
    wav_file.writeframes((audio * 32767).astype("int16").tobytes())
```

For a full example, see `examples/tts.py`.

### Defining Tools

You can give your agent tools to interact with the world. Tools are defined using the `create_tool` helper.

Input and output schemas help the model to correctly pass parameters through to the function. Fury will automatically prune any hallucinated parameters not defined in the input schema.

Learn more in the [OpenAI guide](https://developers.openai.com/api/docs/guides/function-calling/)

```python
from fury import Agent, create_tool

# Define the function
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

If your tool accepts an `emit` parameter, Fury injects a runtime-only callback during execution so the tool can stream structured UI events without exposing `emit` in the model-facing schema.

```python
def search(query: str, emit):
    emit({"id": "search-1", "title": f"Searching for {query}", "type": "tool_call"})
    return {"query": query}
```

These arrive in the chat stream as `event.tool_ui`, separate from `event.tool_call`.

### Coding Assistant Example

Check out `examples/coding-assistant/coding_assistant.py` for a full-featured example that includes:

- File system operations (`read`, `write`, `edit`, `bash`).
- **Skills System**: Loading specialized capabilities from `SKILL.md` files.
- **Memory System**: Using `MEMORY.md` and `SOUL.md` for context.
- **History Manager**: Uses `HistoryManager` to summarize long conversations and save context window.


### TTS Extras
Fury supports both text-to-speech (using NeuTTS Air/Nano) and speech-to-text (using Faster Whisper). 

Install the optional text-to-speech dependencies:

```bash
uv add "fury-sdk[voice,tts]"
```

> Note: `phonemizer` requires the `espeak` system library. On macOS run `brew install espeak`,
> and on Debian/Ubuntu run `sudo apt-get install espeak`.

For local development in this repository:

```bash
uv sync --all-extras
```

## Running Examples

To run the provided examples, ensure you have the package installed.

**Basic Chat:**

```bash
uv run examples/chat.py
```

**Coding Assistant (Based on Pi.dev):**

```bash
uv run examples/coding-assistant/coding_assistant.py
```

**Text-to-Speech (NeuTTS):**

```bash
uv run examples/tts.py
```

**Voice Chat (STT + TTS):**

```bash
uv run examples/voice_chat.py
```

## Project Structure

- `src/agent_lib/`: Core library code.
    - `agent.py`: Main `Agent` class and logic.
- `examples/`: Usage examples.
    - `chat.py`: Basic chat loop.
    - `history_manager.py`: Chat loop with auto-compacting history.
    - `tts.py`: NeuTTS example.
    - `voice_chat.py`: Voice chat with Whisper + NeuTTS.
    - `coding-assistant/`: Advanced agent with file ops and memory.

# Run Tests

To run the pytest tests you will first need to install the additional test deps.
`uv sync --extra test`

Then run:
`uv run pytest -v`
