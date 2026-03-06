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

- **Tool Support**: Define and register custom tools (functions) that the agent can execute.
- **Parallel Tool Execution**: Built-in support for running multiple independent tools in parallel.
- **Image and Voice inputs**: Support for image and voice inputs (using Whisper for STT).
- **Text-to-Speech (TTS)**: Generate audio with NeuTTS via `Agent.speak()`.
- **Streaming Responses**: Real-time streaming of agent responses and reasoning.
- **History Managers**: Use `HistoryManager` for auto-compaction summaries or `StaticHistoryManager` for strict fixed-size context trimming.
- **OpenAI-Standard API Compatible**: Built on top of `AsyncOpenAI`, making it compatible with all OpenAI-API compatible providers. (like Llama.cpp, SGLang, OpenRouter, vLLM, Ollama, etc.).

## Roadmap

- Message regeneration.
- Context-targeting history compaction.

## Installation

Install with uv:

```bash
uv add fury-sdk
```

Install with pip:

```bash
pip install fury-sdk
```

### Examples

If you also want example dependencies:

```bash
uv add "git+https://github.com/huwprosser/fury.git[examples]"
```

### TTS Extras

Install the optional text-to-speech dependencies:

```bash
uv add "fury-sdk[tts]"
```

> Note: `phonemizer` requires the `espeak` system library. On macOS run `brew install espeak`,
> and on Debian/Ubuntu run `sudo apt-get install espeak`.

For local development in this repository:

```bash
uv sync --all-extras
```

## Quick Start

Most basic usage:

```python
from fury import Agent

agent = Agent(
    model="your-model-name",  # e.g., "gpt-4o" or a local model
    system_prompt="You are a helpful assistant.",
    base_url="http://127.0.0.1:8080/v1",  # or https://openrouter.ai/api/v1, https://api.openai.com/v1
    api_key="your-api-key",
)

response = agent.ask("Hello!", history=[])
print(response)
```

### Real Example

Below is a basic chat loop with automatic history-compaction. This allows the bot to keep talking without loosing key facts beyond the context window. It happens automatically and can be configured to suit your models needs.

```python
import asyncio
from fury import Agent, HistoryManager


agent = Agent(
    model="unsloth/GLM-4.6V-Flash-GGUF:Q8_0",
    system_prompt="You are a helpful assistant.",
)

history_manager = HistoryManager(agent=agent)

async def main() -> None:
    while True:
        user_input = input("> ")

        await history_manager.add({"role": "user", "content": user_input})

        buffer = ""

        async for event in agent.chat(history_manager.history):
            if event.content:
                buffer += event.content
                print(event.content, end="", flush=True)

        await history_manager.add({"role": "assistant", "content": buffer})
        print()


if __name__ == "__main__":
    asyncio.run(main())

```

### Static Context Window Example

If you want no summarization and a hard history limit, use `StaticHistoryManager`:

```python
from fury import StaticHistoryManager

history_manager = StaticHistoryManager(
    target_context_length=4096,
    history=[{"role": "system", "content": "You are helpful."}],
)
```

It keeps only the newest messages that fit in the target context length.
See `docs/example.md` for a complete example.

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
def add(a: int, b: int):
    return {"result": a + b}

# Create the tool
add_tool = create_tool(
    id="add",
    description="Add two numbers together",
    execute=add,
    announcement_phrase="Adding numbers...",
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

### Coding Assistant Example

Check out `examples/coding-assistant/coding_assistant.py` for a full-featured example that includes:

- File system operations (`read`, `write`, `edit`, `bash`).
- **Skills System**: Loading specialized capabilities from `SKILL.md` files.
- **Memory System**: Using `MEMORY.md` and `SOUL.md` for context.
- **History Manager**: Uses `HistoryManager` to summarize long conversations and save context window.

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
