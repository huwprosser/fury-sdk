import asyncio

from fury import HistoryManager, StaticHistoryManager


class FakeCompletion:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})]


class FakeChatCompletions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents.pop(0) if self.contents else "summary"
        return FakeCompletion(content)


class FakeClient:
    def __init__(self, contents):
        self.chat = type("Chat", (), {"completions": FakeChatCompletions(contents)})()


def test_history_manager_auto_compacts_when_context_threshold_exceeded():
    client = FakeClient(["Summary content"])
    manager = HistoryManager(
        client=client,
        summary_model="fake-model",
        auto_compact=True,
        context_window=60,
        reserve_tokens=10,
        keep_recent_tokens=10,
        summary_prefix="Summary:",
    )

    async def run():
        for idx in range(6):
            await manager.add({"role": "user", "content": f"hello {'x' * 50} {idx}"})
            await manager.add({"role": "assistant", "content": f"reply {'y' * 50} {idx}"})
        return manager.history

    history = asyncio.run(run())

    assert history[0]["role"] == "system"
    assert history[0]["content"].startswith("Summary:")


def test_history_manager_preserves_recent_messages_after_compaction():
    client = FakeClient(["Summary content"])
    manager = HistoryManager(
        client=client,
        summary_model="fake-model",
        auto_compact=True,
        context_window=60,
        reserve_tokens=10,
        keep_recent_tokens=10,
        summary_prefix="Summary:",
    )

    async def run():
        for idx in range(6):
            await manager.add({"role": "user", "content": f"hello {'x' * 50} {idx}"})
            await manager.add({"role": "assistant", "content": f"reply {'y' * 50} {idx}"})
        return manager.history

    history = asyncio.run(run())

    assert history[-1]["content"].endswith("5")
    assert any(message["content"].endswith("4") for message in history[1:])


def test_history_manager_replaces_existing_summary_instead_of_stacking_multiple_summaries():
    client = FakeClient(["First summary", "Second summary"])
    manager = HistoryManager(
        client=client,
        summary_model="fake-model",
        auto_compact=True,
        context_window=80,
        reserve_tokens=10,
        keep_recent_tokens=10,
        summary_prefix="Summary:",
    )

    first_history = [
        {"role": "user", "content": "u" * 100},
        {"role": "assistant", "content": "a" * 100},
        {"role": "user", "content": "v" * 100},
        {"role": "assistant", "content": "b" * 100},
    ]
    second_history = [
        {"role": "system", "content": "Summary:\nFirst summary"},
        {"role": "user", "content": "c" * 100},
        {"role": "assistant", "content": "d" * 100},
        {"role": "user", "content": "e" * 100},
        {"role": "assistant", "content": "f" * 100},
    ]

    asyncio.run(manager._compact_history(first_history))
    history = asyncio.run(manager._compact_history(second_history))

    assert sum(1 for message in history if message["role"] == "system") == 1
    assert history[0]["content"].startswith("Summary:")
    assert "Second summary" in history[0]["content"]


def test_history_manager_includes_tool_file_ops_in_summary_prompt():
    client = FakeClient(["Summary content"])
    manager = HistoryManager(
        client=client,
        summary_model="fake-model",
        auto_compact=True,
        context_window=40,
        reserve_tokens=10,
        keep_recent_tokens=10,
    )
    history = [
        {
            "role": "assistant",
            "tool_calls": [
                {"name": "read", "arguments": '{"path": "src/app.py"}'},
                {"name": "edit", "arguments": '{"path": "src/app.py"}'},
                {"name": "write", "arguments": '{"path": "README.md"}'},
            ],
        },
        {"role": "user", "content": "x" * 200},
        {"role": "assistant", "content": "y" * 200},
    ]

    asyncio.run(manager._compact_history(history))

    prompt = client.chat.completions.calls[0]["messages"][1]["content"]
    assert "Read files: src/app.py" in prompt
    assert "Modified files: README.md, src/app.py" in prompt


def test_static_history_manager_keeps_latest_messages_within_token_budget():
    manager = StaticHistoryManager(
        target_context_length=8,
        history=[
            {"role": "user", "content": "a" * 16},
            {"role": "assistant", "content": "b" * 16},
        ],
    )

    asyncio.run(manager.add({"role": "user", "content": "c" * 16}))

    assert manager.history == [
        {"role": "assistant", "content": "b" * 16},
        {"role": "user", "content": "c" * 16},
    ]
