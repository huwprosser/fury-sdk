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


if __name__ == "__main__":
    asyncio.run(main())
