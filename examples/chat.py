import asyncio

from fury import Agent, HistoryManager

agent = Agent(
    model="unsloth/Qwen3.5-4B-GGUF:Q4_K_M",
    system_prompt="You are a helpful assistant.",
)

history_manager = HistoryManager(
    agent=agent,
    auto_compact=True,
    context_window=32768,
    reserve_tokens=8192,
    keep_recent_tokens=8000,
)


async def main() -> None:
    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue

        await history_manager.add({"role": "user", "content": user_input})

        buffer = ""
        print()
        async for event in agent.chat(history_manager.history):
            if event.content:
                buffer += event.content
                print(event.content, end="", flush=True)

        await history_manager.add({"role": "assistant", "content": buffer})
        print()


if __name__ == "__main__":
    asyncio.run(main())
