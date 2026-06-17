
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

**Markdown Memory Tool:**

```bash
uv run examples/memory_chat.py
```

**History Compaction:**

```bash
uv run examples/history_compaction.py
```

**Interrupting A Stream With A Hotkey:**

```bash
uv run examples/interruption.py
```

**Coding Assistant (Based on Pi.dev):**

```bash
uv run examples/coding-assistant/coding_assistant.py
```

## Project Structure

- `src/fury/`: Core library code.
    - `agent.py`: Public `Agent` facade and `Runner`.
    - `runtime.py`: Streaming chat loop and interruption handling.
    - `tools.py`: Tool registry and execution.
    - OpenAI SDK client wiring for OpenAI-compatible chat completions.
    - `types.py`: Public event and tool types.
- `examples/`: Usage examples.
    - `chat.py`: Basic chat loop.
    - `memory_chat.py`: Chat loop with a custom tool that edits a markdown memory file.
    - `history_compaction.py`: Generate a compact summary from a list of history messages.
    - `interruption.py`: Chat loop with a hotkey that interrupts a streamed reply and keeps the partial output in history.
    - `coding-assistant/`: Advanced agent with file ops.
