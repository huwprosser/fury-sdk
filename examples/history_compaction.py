import asyncio
import os

import dotenv

from fury import Agent, HistoryCompactor

dotenv.load_dotenv()

agent = Agent(
    base_url=os.getenv("OPENROUTER_BASE_URL", ""),
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
    model="qwen/Qwen3.5-35B-A3B",
    system_prompt="You summarize chat history clearly and concisely.",
)

history = [
    {"role": "user", "content": "I'm building a Python SDK for AI agents."},
    {
        "role": "assistant",
        "content": "We simplified history management and removed persistence.",
    },
    {"role": "user", "content": "I want compaction to be a separate helper module."},
]


async def main() -> None:
    compactor = HistoryCompactor(agent)
    summary = await compactor.compaction_summary(history)
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
