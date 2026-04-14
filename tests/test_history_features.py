import asyncio
import base64
import json
import sys
import tempfile
import types
from pathlib import Path

import pytest

from fury import Agent, HistoryManager, StaticHistoryManager


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


def test_history_manager_persists_raw_messages_to_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = HistoryManager(
            auto_compact=False,
            persist_to_disk=True,
            session_id="session-123",
            history_root=tmpdir,
        )

        asyncio.run(
            manager.extend(
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ]
            )
        )

        assert manager.history_path is not None
        persisted = [
            json.loads(line)
            for line in manager.history_path.read_text(encoding="utf-8").splitlines()
        ]
        assert persisted == manager.history


def test_history_manager_loads_persisted_history_on_init():
    with tempfile.TemporaryDirectory() as tmpdir:
        first = HistoryManager(
            auto_compact=False,
            persist_to_disk=True,
            session_id="session-123",
            history_root=tmpdir,
        )
        asyncio.run(first.add({"role": "user", "content": "hello"}))
        asyncio.run(first.add({"role": "assistant", "content": "world"}))

        second = HistoryManager(
            auto_compact=False,
            persist_to_disk=True,
            session_id="session-123",
            history_root=tmpdir,
        )

        assert second.history == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]


def test_history_manager_persists_direct_history_appends():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = HistoryManager(
            auto_compact=False,
            persist_to_disk=True,
            session_id="session-123",
            history_root=tmpdir,
        )

        manager.history.append({"role": "assistant", "content": "partial reply"})

        persisted = [
            json.loads(line)
            for line in manager.history_path.read_text(encoding="utf-8").splitlines()
        ]
        assert persisted == [{"role": "assistant", "content": "partial reply"}]


def test_history_manager_compacts_loaded_history_before_first_new_message():
    with tempfile.TemporaryDirectory() as tmpdir:
        seed = HistoryManager(
            auto_compact=False,
            persist_to_disk=True,
            session_id="session-123",
            history_root=tmpdir,
        )

        async def seed_history():
            for idx in range(6):
                await seed.add({"role": "user", "content": f"hello {'x' * 50} {idx}"})
                await seed.add(
                    {"role": "assistant", "content": f"reply {'y' * 50} {idx}"}
                )

        asyncio.run(seed_history())

        client = FakeClient(["Summary content"])
        manager = HistoryManager(
            client=client,
            summary_model="fake-model",
            auto_compact=True,
            context_window=60,
            reserve_tokens=10,
            keep_recent_tokens=10,
            summary_prefix="Summary:",
            persist_to_disk=True,
            session_id="session-123",
            history_root=tmpdir,
        )

        asyncio.run(manager.add({"role": "user", "content": "fresh message"}))

        assert manager.history[0]["role"] == "system"
        assert manager.history[0]["content"].startswith("Summary:")
        assert manager.history[-1] == {"role": "user", "content": "fresh message"}
        persisted = [
            json.loads(line)
            for line in manager.history_path.read_text(encoding="utf-8").splitlines()
        ]
        assert len(persisted) == 13
        assert all(message["role"] != "system" for message in persisted)
        assert persisted[-1] == {"role": "user", "content": "fresh message"}


def test_history_manager_prints_notice_when_auto_compacting(capfd):
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

    asyncio.run(run())
    captured = capfd.readouterr()

    assert "[history] Compacting history" in captured.out


def test_history_manager_suppresses_compaction_notice_with_suppress_logs(capfd):
    client = FakeClient(["Summary content"])
    agent = Agent(
        model="test-model",
        system_prompt="You are helpful.",
        disable_stt=True,
        suppress_logs=True,
    )
    agent.client = client

    manager = HistoryManager(
        agent=agent,
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

    asyncio.run(run())
    captured = capfd.readouterr()

    assert captured.out == ""


def test_history_manager_persists_raw_messages_even_when_compacting():
    client = FakeClient(["Summary content"])

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = HistoryManager(
            client=client,
            summary_model="fake-model",
            auto_compact=True,
            context_window=60,
            reserve_tokens=10,
            keep_recent_tokens=10,
            summary_prefix="Summary:",
            persist_to_disk=True,
            session_id="session-123",
            history_root=tmpdir,
        )

        async def run():
            for idx in range(6):
                await manager.add({"role": "user", "content": f"hello {'x' * 50} {idx}"})
                await manager.add(
                    {"role": "assistant", "content": f"reply {'y' * 50} {idx}"}
                )

        asyncio.run(run())

        assert manager.history[0]["role"] == "system"
        persisted = [
            json.loads(line)
            for line in manager.history_path.read_text(encoding="utf-8").splitlines()
        ]
        assert len(persisted) == 12
        assert all(message["role"] != "system" for message in persisted)
        assert persisted[0]["content"].endswith("0")
        assert persisted[-1]["content"].endswith("5")


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


def test_history_manager_accepts_agent_as_first_positional_argument():
    agent = Agent(
        model="test-model",
        system_prompt="You are helpful.",
        disable_stt=True,
        suppress_logs=True,
    )
    manager = HistoryManager(agent, auto_compact=False)

    assert manager.agent is agent
    assert manager.history == []


def test_history_manager_requires_session_id_for_disk_persistence():
    with pytest.raises(ValueError, match="session_id is required"):
        HistoryManager(auto_compact=False, persist_to_disk=True)


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


def test_history_manager_add_image_appends_valid_user_message():
    manager = HistoryManager(auto_compact=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "sample.jpg"
        image_path.write_bytes(b"fake-image")

        history = asyncio.run(manager.add_image(str(image_path), text="Describe this"))

    assert history[-1]["role"] == "user"
    assert history[-1]["content"][0] == {"type": "text", "text": "Describe this"}
    assert history[-1]["content"][1]["type"] == "image_url"
    assert history[-1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_history_manager_add_voice_requires_agent():
    manager = HistoryManager(auto_compact=False)

    try:
        asyncio.run(manager.add_voice(base64.b64encode(b"fake").decode("utf-8")))
    except ValueError as exc:
        assert "requires an Agent instance" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_history_manager_prewarms_voice_model_when_available(monkeypatch):
    init_calls = []

    class FakeWhisperModel:
        pass

    faster_whisper_module = types.ModuleType("faster_whisper")
    faster_whisper_module.WhisperModel = (
        lambda name: init_calls.append(name) or FakeWhisperModel()
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper_module)

    agent = Agent(model="test-model", system_prompt="You are helpful.")

    assert agent.stt is None

    HistoryManager(agent=agent, auto_compact=False)

    assert isinstance(agent.stt, FakeWhisperModel)
    assert init_calls == ["base.en"]


def test_history_manager_skips_voice_prewarm_when_stt_disabled(monkeypatch):
    init_calls = []

    class FakeWhisperModel:
        pass

    faster_whisper_module = types.ModuleType("faster_whisper")
    faster_whisper_module.WhisperModel = (
        lambda name: init_calls.append(name) or FakeWhisperModel()
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper_module)

    agent = Agent(
        model="test-model",
        system_prompt="You are helpful.",
        disable_stt=True,
    )

    HistoryManager(agent=agent, auto_compact=False)

    assert agent.stt is None
    assert init_calls == []


def test_history_manager_ignores_transcription_prewarm_failures(monkeypatch):
    faster_whisper_module = types.ModuleType("faster_whisper")
    faster_whisper_module.WhisperModel = lambda _name: (_ for _ in ()).throw(
        RuntimeError("boom")
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper_module)

    agent = Agent(model="test-model", system_prompt="You are helpful.")

    manager = HistoryManager(agent=agent, auto_compact=False)

    assert manager.agent is agent
    assert agent.stt is None


def test_history_manager_add_voice_appends_transcribed_user_message(monkeypatch):
    agent = Agent(model="test-model", system_prompt="You are helpful.")
    manager = HistoryManager(agent=agent, auto_compact=False)

    class FakeWhisper:
        def transcribe(self, audio):
            return [type("Seg", (), {"text": " hello "})()], None

    monkeypatch.setattr(agent, "stt", FakeWhisper())

    audio_module = types.ModuleType("fury.utils.audio")
    audio_module.load_audio = lambda *_args, **_kwargs: (b"audio", 16000)
    monkeypatch.setitem(sys.modules, "fury.utils.audio", audio_module)

    history = asyncio.run(manager.add_voice(base64.b64encode(b"fake").decode("utf-8")))

    assert history[-1] == {"role": "user", "content": "hello"}


def test_history_manager_add_voice_raises_when_stt_disabled():
    agent = Agent(
        model="test-model",
        system_prompt="You are helpful.",
        disable_stt=True,
    )
    manager = HistoryManager(agent=agent, auto_compact=False)

    with pytest.raises(RuntimeError, match="STT is disabled"):
        asyncio.run(manager.add_voice(base64.b64encode(b"fake").decode("utf-8")))
