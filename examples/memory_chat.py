import asyncio
from pathlib import Path

from fury import Agent, HistoryManager, Tool

MODEL_NAME = "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_X"
MEMORY_PATH = Path(".fury/memory.md")

BASE_SYSTEM_PROMPT = """You are a helpful assistant.

You have a markdown memory file included below. Use it as durable context.
When the user asks you to remember, forget, or update durable information, call
edit_memory with the full replacement markdown content.
"""


def read_memory() -> str:
    if not MEMORY_PATH.exists():
        return "# Memory\n\n"
    return MEMORY_PATH.read_text(encoding="utf-8")


def build_system_prompt() -> str:
    return f"{BASE_SYSTEM_PROMPT}\n\n<memory>\n{read_memory()}\n</memory>"


def edit_memory(content: str) -> dict:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(content, encoding="utf-8")
    return {"path": str(MEMORY_PATH), "content": content}


edit_memory_tool = Tool(
    name="edit_memory",
    description="Replace the durable markdown memory file with new markdown content.",
    execute=edit_memory,
    input_schema={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The full replacement markdown memory content.",
            }
        },
        "required": ["content"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
)

agent = Agent(
    model=MODEL_NAME,
    system_prompt=build_system_prompt(),
    tools=[edit_memory_tool],
)
history_manager = HistoryManager(target_context_length=32768)


async def main() -> None:
    print(f"[memory file: {MEMORY_PATH}]")
    print("[commands: /memory, /exit]")

    while True:
        print()
        user_input = input("> ").strip()

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            break
        if user_input == "/memory":
            print(read_memory())
            continue

        await history_manager.add({"role": "user", "content": user_input})

        transcript = []
        runner = agent.runner()
        async for event in runner.chat(history_manager.history):
            if event.tool_ui:
                print(f"\n[tool] {event.tool_ui.title}")
            if event.content:
                print(event.content, end="", flush=True)
            if event.history_delta:
                transcript.append(event.history_delta.message)

        print()
        await history_manager.extend(transcript)


if __name__ == "__main__":
    asyncio.run(main())
