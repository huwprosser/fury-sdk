import asyncio
import base64
import json
import sys
import tempfile
import types
from pathlib import Path

import pytest
from conftest import (
    FakeCompletion as StreamFakeCompletion,
    FakeDelta,
    SequencedCreate,
    make_fake_client,
)

from fury import Agent, HistoryManager, StaticHistoryManager
from fury.historymanager import (
    HISTORY_ACTIVE_VARIANT_ID_KEY,
    HISTORY_MESSAGE_ID_KEY,
    HISTORY_MESSAGE_VARIANTS_KEY,
)
from fury.multimodal import IMAGE_HISTORY_PLACEHOLDER, materialize_history_message


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


def strip_history_ids(history):
    return [
        {
            key: value
            for key, value in message.items()
            if key != HISTORY_MESSAGE_ID_KEY
        }
        for message in history
    ]


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
        assert all(message[HISTORY_MESSAGE_ID_KEY] for message in persisted)


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

        assert strip_history_ids(second.history) == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        assert [message[HISTORY_MESSAGE_ID_KEY] for message in second.history] == [
            message[HISTORY_MESSAGE_ID_KEY] for message in first.history
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
        assert strip_history_ids(persisted) == [
            {"role": "assistant", "content": "partial reply"}
        ]
        assert persisted[0][HISTORY_MESSAGE_ID_KEY]


def test_history_manager_rewrites_missing_ids_when_loading_persisted_history():
    with tempfile.TemporaryDirectory() as tmpdir:
        first = HistoryManager(
            auto_compact=False,
            persist_to_disk=True,
            session_id="session-123",
            history_root=tmpdir,
        )
        assert first.history_path is not None
        first.history_path.parent.mkdir(parents=True, exist_ok=True)
        first.history_path.write_text(
            json.dumps({"role": "user", "content": "legacy"}) + "\n",
            encoding="utf-8",
        )

        manager = HistoryManager(
            auto_compact=False,
            persist_to_disk=True,
            session_id="session-123",
            history_root=tmpdir,
        )

        assert manager.history[0][HISTORY_MESSAGE_ID_KEY]
        persisted = [
            json.loads(line)
            for line in manager.history_path.read_text(encoding="utf-8").splitlines()
        ]
        assert persisted == manager.history


def test_history_manager_migrates_legacy_fury_ids_when_loading_persisted_history():
    with tempfile.TemporaryDirectory() as tmpdir:
        first = HistoryManager(
            auto_compact=False,
            persist_to_disk=True,
            session_id="session-123",
            history_root=tmpdir,
        )
        assert first.history_path is not None
        first.history_path.parent.mkdir(parents=True, exist_ok=True)
        first.history_path.write_text(
            json.dumps(
                {"role": "user", "content": "legacy", "_fury_id": "legacy-id"}
            )
            + "\n",
            encoding="utf-8",
        )

        manager = HistoryManager(
            auto_compact=False,
            persist_to_disk=True,
            session_id="session-123",
            history_root=tmpdir,
        )

        assert manager.history[0][HISTORY_MESSAGE_ID_KEY] == "legacy-id"
        assert "_fury_id" not in manager.history[0]
        persisted = [
            json.loads(line)
            for line in manager.history_path.read_text(encoding="utf-8").splitlines()
        ]
        assert persisted == manager.history


def test_history_manager_edits_message_by_id_and_rewrites_persisted_history():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = HistoryManager(
            auto_compact=False,
            persist_to_disk=True,
            session_id="session-123",
            history_root=tmpdir,
        )
        asyncio.run(manager.add({"role": "user", "content": "hello"}))
        message_id = manager.history[0][HISTORY_MESSAGE_ID_KEY]

        manager.edit_message(message_id, {"role": "user", "content": "edited"})

        assert manager.history == [
            {"role": "user", "content": "edited", HISTORY_MESSAGE_ID_KEY: message_id}
        ]
        persisted = [
            json.loads(line)
            for line in manager.history_path.read_text(encoding="utf-8").splitlines()
        ]
        assert persisted == manager.history


def test_history_manager_regenerates_message_as_active_variant():
    create = SequencedCreate(
        [StreamFakeCompletion([FakeDelta(content="new"), FakeDelta(content=" answer")])]
    )
    agent = Agent(
        model="test-model",
        system_prompt="You are helpful.",
        disable_stt=True,
        suppress_logs=True,
    )
    agent.client = make_fake_client(create)
    manager = HistoryManager(agent=agent, auto_compact=False)
    asyncio.run(
        manager.extend(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "old answer"},
            ]
        )
    )
    message_id = manager.history[1][HISTORY_MESSAGE_ID_KEY]

    asyncio.run(manager.regenerate_message(message_id))

    message = manager.history[1]
    variants = message[HISTORY_MESSAGE_VARIANTS_KEY]
    assert message["content"] == "new answer"
    assert message[HISTORY_ACTIVE_VARIANT_ID_KEY] == variants[1]["id"]
    assert [variant["message"]["content"] for variant in variants] == [
        "old answer",
        "new answer",
    ]
    assert create.calls[0]["messages"][-1] == {
        "role": "user",
        "content": "question",
    }
    assert all(HISTORY_MESSAGE_ID_KEY not in msg for msg in create.calls[0]["messages"])


def test_history_manager_set_variant_swaps_active_message_content():
    create = SequencedCreate(
        [StreamFakeCompletion([FakeDelta(content="new answer")])]
    )
    agent = Agent(
        model="test-model",
        system_prompt="You are helpful.",
        disable_stt=True,
        suppress_logs=True,
    )
    agent.client = make_fake_client(create)
    manager = HistoryManager(agent=agent, auto_compact=False)
    asyncio.run(
        manager.extend(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "old answer"},
            ]
        )
    )
    message_id = manager.history[1][HISTORY_MESSAGE_ID_KEY]
    asyncio.run(manager.regenerate_message(message_id))
    original_variant_id = manager.history[1][HISTORY_MESSAGE_VARIANTS_KEY][0]["id"]

    manager.set_variant(message_id, original_variant_id)

    assert manager.history[1]["content"] == "old answer"
    assert manager.history[1][HISTORY_ACTIVE_VARIANT_ID_KEY] == original_variant_id


def test_history_manager_regenerate_requires_assistant_message():
    agent = Agent(
        model="test-model",
        system_prompt="You are helpful.",
        disable_stt=True,
        suppress_logs=True,
    )
    manager = HistoryManager(agent=agent, auto_compact=False)
    asyncio.run(manager.add({"role": "user", "content": "question"}))
    message_id = manager.history[0][HISTORY_MESSAGE_ID_KEY]

    with pytest.raises(ValueError, match="Only assistant messages"):
        asyncio.run(manager.regenerate_message(message_id))


def test_history_manager_delete_message_by_id_and_rewrites_persisted_history():
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
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "second"},
                ]
            )
        )
        first_id = manager.history[0][HISTORY_MESSAGE_ID_KEY]

        manager.delete_message(first_id)

        assert strip_history_ids(manager.history) == [
            {"role": "assistant", "content": "second"}
        ]
        persisted = [
            json.loads(line)
            for line in manager.history_path.read_text(encoding="utf-8").splitlines()
        ]
        assert persisted == manager.history


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
        assert manager.history[-1]["role"] == "user"
        assert manager.history[-1]["content"] == "fresh message"
        persisted = [
            json.loads(line)
            for line in manager.history_path.read_text(encoding="utf-8").splitlines()
        ]
        assert len(persisted) == 13
        assert all(message["role"] != "system" for message in persisted)
        assert persisted[-1]["role"] == "user"
        assert persisted[-1]["content"] == "fresh message"


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

    assert strip_history_ids(manager.history) == [
        {"role": "assistant", "content": "b" * 16},
        {"role": "user", "content": "c" * 16},
    ]


def test_history_manager_add_image_stores_placeholder_by_default():
    manager = HistoryManager(auto_compact=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "sample.jpg"
        image_path.write_bytes(b"fake-image")

        history = asyncio.run(manager.add_image(str(image_path), text="Describe this"))
        materialized = materialize_history_message(history[-1])

    assert history[-1]["role"] == "user"
    assert history[-1]["content"][0] == {"type": "text", "text": "Describe this"}
    assert history[-1]["content"][1] == {
        "type": "text",
        "text": IMAGE_HISTORY_PLACEHOLDER,
    }
    assert history[-1]["_fury_multimodal"]["kind"] == "image_path"
    assert materialized["content"][1]["type"] == "image_url"
    assert materialized["content"][1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )


def test_materialized_history_message_strips_internal_history_id():
    manager = HistoryManager(auto_compact=False)
    asyncio.run(manager.add({"role": "user", "content": "hello"}))

    materialized = materialize_history_message(manager.history[0])

    assert materialized == {"role": "user", "content": "hello"}


def test_history_manager_add_image_saves_raw_image_when_enabled():
    manager = HistoryManager(auto_compact=False, save_images_to_history=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "sample.jpg"
        image_path.write_bytes(b"fake-image")

        history = asyncio.run(manager.add_image(str(image_path), text="Describe this"))

    assert history[-1]["role"] == "user"
    assert history[-1]["content"][1]["type"] == "image_url"
    assert history[-1]["content"][1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )
    assert "_fury_multimodal" not in history[-1]


def test_history_manager_persisted_image_history_avoids_base64_by_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = HistoryManager(
            auto_compact=False,
            persist_to_disk=True,
            session_id="session-123",
            history_root=tmpdir,
        )
        image_path = Path(tmpdir) / "sample.jpg"
        image_path.write_bytes(b"fake-image")

        asyncio.run(manager.add_image(str(image_path), text="Describe this"))

        persisted = manager.history_path.read_text(encoding="utf-8")

    assert IMAGE_HISTORY_PLACEHOLDER in persisted
    assert "base64," not in persisted


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

    assert history[-1]["role"] == "user"
    assert history[-1]["content"] == "hello"


def test_history_manager_add_voice_raises_when_stt_disabled():
    agent = Agent(
        model="test-model",
        system_prompt="You are helpful.",
        disable_stt=True,
    )
    manager = HistoryManager(agent=agent, auto_compact=False)

    with pytest.raises(RuntimeError, match="STT is disabled"):
        asyncio.run(manager.add_voice(base64.b64encode(b"fake").decode("utf-8")))
