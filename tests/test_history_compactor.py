import asyncio

from conftest import FakeCompletion, FakeDelta, SequencedCreate, make_fake_client

from fury import Agent, DEFAULT_COMPACTION_PROMPT, HistoryCompactor


def test_history_compactor_returns_summary_string():
    create = SequencedCreate([FakeCompletion([FakeDelta(content="summary")])])
    agent = Agent(model="test-model", system_prompt="You summarize chats.")
    agent.client = make_fake_client(create)
    compactor = HistoryCompactor(agent)

    result = asyncio.run(
        compactor.compact([
            {"role": "user", "content": "I want to build a CLI."},
            {"role": "assistant", "content": "Use argparse."},
        ])
    )

    assert result == "summary"
    assert create.calls[0]["messages"][-1] == {
        "role": "user",
        "content": DEFAULT_COMPACTION_PROMPT,
    }


def test_history_compactor_accepts_custom_prompt_and_model_override():
    create = SequencedCreate([FakeCompletion([FakeDelta(content=" custom summary ")])])
    agent = Agent(model="default-model", system_prompt="You summarize chats.")
    agent.client = make_fake_client(create)
    compactor = HistoryCompactor(agent, prompt="default prompt")

    result = asyncio.run(
        compactor.compaction_summary(
            [{"role": "user", "content": "hello"}],
            prompt="custom prompt",
            model="summary-model",
        )
    )

    assert result == "custom summary"
    assert create.calls[0]["model"] == "summary-model"
    assert create.calls[0]["messages"][-1] == {
        "role": "user",
        "content": "custom prompt",
    }
