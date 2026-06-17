import asyncio
import os

import dotenv

from fury import Agent, HistoryManager

dotenv.load_dotenv()

agent = Agent(
    base_url=os.getenv("OPENROUTER_BASE_URL", ""),
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
    model="qwen/Qwen3.5-35B-A3B",
    system_prompt="You are a helpful assistant.",
    generation_params={
        "extra_body": {
            "reasoning": {
                "enabled": True,
            }
        }
    },
)

history_manager = HistoryManager(target_context_length=32768)


async def main() -> None:
    while True:
        print()
        user_input = input("> ")

        await history_manager.add({"role": "user", "content": user_input})

        transcript = []

        runner = agent.runner()
        async for event in runner.chat(history_manager.history, reasoning=True):
            if event.content:
                print(event.content, end="", flush=True)
            if event.history_delta:
                transcript.append(event.history_delta.message)
            if event.reasoning:
                print(event.reasoning, end="", flush=True)

        await history_manager.extend(transcript)


if __name__ == "__main__":
    asyncio.run(main())
