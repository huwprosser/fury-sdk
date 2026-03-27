
## Running Examples

To run the provided examples, ensure you have the package installed.

Add dependencies:

```bash
uv add "git+https://github.com/huwprosser/fury.git[examples]"
```

**Basic Chat:**

```bash
uv run examples/chat.py
```

**Chat With Durable Memory:**

```bash
uv run examples/memory_chat.py
```

**Interrupting A Stream With A Hotkey:**

```bash
uv run examples/interruption.py
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

- `src/fury/`: Core library code.
    - `agent.py`: Public `Agent` facade and `Runner`.
    - `runtime.py`: Streaming chat loop and interruption handling.
    - `tools.py`: Tool registry and execution.
    - `transport.py`: Minimal OpenAI-compatible transport client.
    - `types.py`: Public event and tool types.
- `examples/`: Usage examples.
    - `chat.py`: Basic chat loop.
    - `memory_chat.py`: Chat loop with a named durable-memory scope and a scoped memory tool.
    - `interruption.py`: Chat loop with a hotkey that interrupts a streamed reply and keeps the partial output in history.
    - `tts.py`: NeuTTS example.
    - `voice_chat.py`: Voice chat with Whisper + NeuTTS.
    - `coding-assistant/`: Advanced agent with file ops and memory.
