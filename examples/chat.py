import asyncio

from fury import Agent, HistoryManager

agent = Agent(
    model="unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_X",
    system_prompt="You are a helpful assistant.",
)

history_manager = HistoryManager(agent=agent)


async def main() -> None:
    while True:
        print()
        user_input = input("> ")

        await history_manager.add({"role": "user", "content": user_input})

        buffer = ""

        runner = agent.runner()
        async for event in runner.chat(history_manager.history):
            if event.content:
                buffer += event.content
                print(event.content, end="", flush=True)

        await history_manager.add({"role": "assistant", "content": buffer})


if __name__ == "__main__":
    asyncio.run(main())
