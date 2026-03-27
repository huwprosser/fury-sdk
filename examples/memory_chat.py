import asyncio

from fury import Agent, HistoryManager, MemoryStore

MODEL_NAME = "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_X"
MEMORY_SCOPE = "chatbot-demo"
MEMORY_ROOT = ".fury/memory"

SYSTEM_PROMPT = """You are a helpful assistant.

Use the memory tool for durable user preferences and facts that will matter in future turns.
Do not store short-lived conversational details or temporary planning notes.
"""

memory_store = MemoryStore(MEMORY_ROOT)
agent = Agent(
    model=MODEL_NAME,
    system_prompt=SYSTEM_PROMPT,
    memory_store=memory_store,
    memory_scope=MEMORY_SCOPE,
)
history_manager = HistoryManager(agent=agent)


def print_memory() -> None:
    store = memory_store.get_scope(MEMORY_SCOPE)
    entries = store["entries"]
    print()
    print(f"[memory scope: {MEMORY_SCOPE}]")
    if not entries:
        print("(empty)")
        return

    for entry in entries:
        prefix = "*" if entry.get("pinned") else "-"
        print(f"{prefix} {entry['id']}: {entry['content']}")


async def main() -> None:
    print(f"[memory root: {MEMORY_ROOT}]")
    print(f"[memory scope: {MEMORY_SCOPE}]")
    print("[commands: /memory, /remember <text>, /forget <text>, /exit]")

    while True:
        print()
        user_input = input("> ").strip()

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            break
        if user_input == "/memory":
            print_memory()
            continue
        if user_input.startswith("/remember "):
            result = memory_store.add(
                scope=MEMORY_SCOPE,
                content=user_input[len("/remember ") :].strip(),
                source="user",
            )
            print(result.get("error") or result.get("message") or "Stored.")
            continue
        if user_input.startswith("/forget "):
            result = memory_store.remove(
                scope=MEMORY_SCOPE,
                match_text=user_input[len("/forget ") :].strip(),
            )
            print(result.get("error") or "Removed.")
            continue

        await history_manager.add({"role": "user", "content": user_input})

        buffer = ""
        last_stream_kind = None
        runner = agent.runner()
        async for event in runner.chat(history_manager.history):
            if event.tool_ui:
                if last_stream_kind == "chunk":
                    print()
                last_stream_kind = "tool_ui"
                print(f"[tool] {event.tool_ui.title}")

            if event.content:
                if last_stream_kind == "tool_ui":
                    last_stream_kind = None
                buffer += event.content
                print(event.content, end="", flush=True)
                last_stream_kind = "chunk"

        print()
        await history_manager.add({"role": "assistant", "content": buffer})


if __name__ == "__main__":
    asyncio.run(main())
