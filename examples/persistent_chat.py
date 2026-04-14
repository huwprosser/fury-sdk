import asyncio

from fury import Agent, HistoryManager

MODEL_NAME = "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_X"
SESSION_ID = "demo-session"

agent = Agent(
    model=MODEL_NAME,
    system_prompt="You are a helpful assistant.",
)

history_manager = HistoryManager(
    agent,
    persist_to_disk=True,
    session_id=SESSION_ID,
)


async def main() -> None:
    restored_messages = len(history_manager.history)
    print(f"[session: {SESSION_ID}]")
    if history_manager.history_path is not None:
        print(f"[history file: {history_manager.history_path}]")
    print(f"[restored messages: {restored_messages}]")
    print("[commands: /exit, /quit]")

    while True:
        print()
        user_input = input("> ").strip()

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            break

        await history_manager.add({"role": "user", "content": user_input})

        buffer = ""
        runner = agent.runner()
        async for event in runner.chat(history_manager.history):
            if event.content:
                buffer += event.content
                print(event.content, end="", flush=True)

        print()
        await history_manager.add({"role": "assistant", "content": buffer})


if __name__ == "__main__":
    asyncio.run(main())
