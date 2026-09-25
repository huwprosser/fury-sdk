import asyncio
import json
from contextlib import aclosing

import pytest

from fury.sessions import (
    CallbackStorage, InvalidSessionError, JSONLStorage, SessionExistsError,
    SessionManager, SessionNotFoundError, SessionRecord,
)
from fury.types import ChatStreamEvent, HistoryDelta


def run(coro):
    return asyncio.run(coro)


def test_jsonl_roundtrip_metadata_and_context(tmp_path):
    async def check():
        storage = JSONLStorage(tmp_path)
        session = await SessionManager.create(storage=storage, id="abc", metadata={"user": "1"})
        await session.append_message({"role": "user", "content": "héllo", "extra": [1]})
        await session.append_custom("app.state", {"step": 3})
        await session.update_metadata({"title": "Trip", "nested": {"a": 1}, "null": None})
        await session.update_metadata({"nested": {"b": 2}})
        await session.delete_metadata("title")
        restored = await SessionManager.open(storage=storage, id="abc")
        assert restored.metadata == {"user": "1", "nested": {"b": 2}, "null": None}
        assert restored.build_context() == [{"role": "user", "content": "héllo", "extra": [1]}]
        restored.entries[0]["message"]["extra"].append(2)
        restored.metadata["nested"]["b"] = 4
        assert restored.build_context()[0]["extra"] == [1]
        assert restored.metadata["nested"]["b"] == 2
        assert (await storage.list())[0]["id"] == "abc"
        with pytest.raises(SessionExistsError):
            await SessionManager.create(storage=storage, id="abc")
        await storage.delete("abc")
        with pytest.raises(SessionNotFoundError):
            await storage.read("abc")
    run(check())


def test_promotion_compaction_and_legacy_format(tmp_path):
    async def check():
        session = SessionManager.in_memory(id="legacy")
        await session.append_message({"role": "user", "content": "old"})
        tail = [{"role": "user", "content": "recent"}]
        await session.append_compaction("summary", 123, tail)
        await session.append_message({"role": "assistant", "content": "new"})
        assert not session.is_persisted()
        storage = JSONLStorage(tmp_path)
        await session.promote(storage=storage)
        path = next(tmp_path.glob("*.jsonl"))
        lines = path.read_text().splitlines()
        header = json.loads(lines[0])
        del header["metadata"]  # Existing voice-agent format.
        path.write_text(json.dumps(header) + "\n" + "\n".join(lines[1:]) + "\n")
        restored = await SessionManager.load_or_create(storage=storage, id="legacy")
        assert restored.build_context() == [
            {"role": "user", "content": "[Previous conversation summary]\nsummary"},
            *tail, {"role": "assistant", "content": "new"},
        ]
        assert len(restored.entries) == 3
    run(check())


@pytest.mark.parametrize("suffix", [b'{"type":', b'\xff'])
def test_torn_tail_repaired_before_append(tmp_path, suffix):
    async def check():
        storage = JSONLStorage(tmp_path)
        await SessionManager.create(storage=storage, id="torn")
        path = next(tmp_path.glob("*.jsonl"))
        with path.open("ab") as handle:
            handle.write(suffix)
        restored = await SessionManager.open(storage=storage, id="torn")
        await restored.append_custom("recovered")
        assert len((await storage.read("torn")).entries) == 1
        for line in path.read_text().splitlines():
            json.loads(line)
    run(check())


def test_corruption_not_replaced(tmp_path):
    async def check():
        storage = JSONLStorage(tmp_path)
        await SessionManager.create(storage=storage, id="bad")
        path = next(tmp_path.glob("*.jsonl"))
        original = path.read_bytes() + b'invalid\n'
        path.write_bytes(original)
        with pytest.raises(InvalidSessionError):
            await SessionManager.load_or_create(storage=storage, id="bad")
        assert path.read_bytes() == original
    run(check())


@pytest.mark.parametrize("id", ["../escape", "", "a/b", "*", "a" * 129])
def test_ids_validated(id):
    with pytest.raises(ValueError):
        SessionManager.in_memory(id=id)


def test_callback_backend_and_failure():
    async def check():
        records = {}
        async def create(header):
            if header["id"] in records:
                raise SessionExistsError(header["id"])
            records[header["id"]] = SessionRecord(header, [])
        async def read(id):
            if id not in records:
                raise SessionNotFoundError(id)
            return records[id]
        async def append(id, entries):
            records[id].entries.extend(entries)
        storage = CallbackStorage(create=create, read=read, append=append)
        session = await SessionManager.load_or_create(storage=storage, id="db")
        await session.append_custom("event", [1, True, None])
        assert len(records["db"].entries) == 1
        async def fail(*args):
            raise OSError("offline")
        storage.append = fail
        with pytest.raises(OSError):
            await session.append_custom("lost")
        assert len(session.entries) == 1
        with pytest.raises(ValueError):
            await session.append_custom("nan", float("nan"))
        with pytest.raises(TypeError):
            await session.update_metadata({1: "invalid"})
    run(check())


def test_cancellation_finishes_write_before_unlock():
    async def check():
        started, release = asyncio.Event(), asyncio.Event()
        entries = []
        async def create(header):
            pass
        async def read(id):
            raise SessionNotFoundError(id)
        async def append(id, batch):
            started.set()
            await release.wait()
            entries.extend(batch)
        session = await SessionManager.create(
            storage=CallbackStorage(create=create, read=read, append=append), id="cancel")
        task = asyncio.create_task(session.append_custom("written"))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert session.entries == entries
        assert len(entries) == 1
    run(check())


class FakeRunner:
    interrupted = False
    partial_response = ""

    def __init__(self, interrupt=False):
        self.should_interrupt = interrupt
        self.closed = False

    async def chat(self, history, **options):
        try:
            self.partial_response = "Checking."
            yield ChatStreamEvent(content="Checking.")
            yield ChatStreamEvent(history_delta=HistoryDelta(
                kind="assistant_tool_calls", message={"role": "assistant", "content": "Checking."}))
            self.partial_response += "Partial"
            self.interrupted = self.should_interrupt
            yield ChatStreamEvent(content="Partial")
            if not self.interrupted:
                yield ChatStreamEvent(history_delta=HistoryDelta(
                    kind="assistant_final", message={"role": "assistant", "content": "Partial"}))
        finally:
            self.closed = True


@pytest.mark.parametrize("interrupt", [False, True])
def test_stream_records_without_duplicate_narration(interrupt):
    async def check():
        session = SessionManager.in_memory()
        runner = FakeRunner(interrupt)
        events = [e async for e in session.stream(runner, "Hello")]
        assert events
        assert [m["content"] for m in session.build_context()] == ["Hello", "Checking.", "Partial"]
        assert runner.closed
    run(check())


@pytest.mark.parametrize("mode", ["interrupt", "cancel", "complete"])
def test_real_runner_persistence(tmp_path, mode):
    from conftest import FakeCompletion, FakeDelta, SequencedCreate, make_fake_client
    from fury import Agent

    async def check():
        agent = Agent(model="test", system_prompt="", suppress_logs=True)
        await agent.client.close()
        completion = FakeCompletion([FakeDelta(content="Hello"), FakeDelta(content=" world")])
        agent.client = make_fake_client(SequencedCreate([completion]))
        storage = JSONLStorage(tmp_path)
        session = await SessionManager.create(storage=storage, id="real")
        runner = agent.runner()
        async for event in session.stream(runner, "Hi"):
            if event.content == "Hello" and mode != "complete":
                getattr(runner, mode)()
        restored = await SessionManager.open(storage=storage, id="real")
        contents = [m["content"] for m in restored.build_context()]
        assert contents == {
            "interrupt": ["Hi", "Hello"],
            "cancel": ["Hi"],
            "complete": ["Hi", "Hello world"],
        }[mode]
    run(check())


def test_failed_promotion_requires_recovery():
    async def check():
        async def create(header):
            pass
        async def read(id):
            raise SessionNotFoundError(id)
        async def append(id, entries):
            raise OSError("offline")
        session = SessionManager.in_memory()
        await session.append_custom("existing")
        storage = CallbackStorage(create=create, read=read, append=append)
        with pytest.raises(OSError):
            await session.promote(storage=storage)
        with pytest.raises(RuntimeError, match="Promotion failed"):
            await session.append_custom("new")
        with pytest.raises(RuntimeError, match="Promotion failed"):
            await session.promote(storage=storage)
    run(check())


@pytest.mark.parametrize("field,value", [("version", 2), ("type", "other"), ("id", "wrong")])
def test_invalid_header_rejected(tmp_path, field, value):
    async def check():
        storage = JSONLStorage(tmp_path)
        await SessionManager.create(storage=storage, id="header")
        path = next(tmp_path.glob("*.jsonl"))
        header = json.loads(path.read_text())
        header[field] = value
        path.write_text(json.dumps(header) + "\n")
        with pytest.raises(InvalidSessionError):
            await SessionManager.open(storage=storage, id="header")
    run(check())


def test_early_close_discards_uncommitted_text():
    async def check():
        session = SessionManager.in_memory()
        runner = FakeRunner()
        async with aclosing(session.stream(runner, "Hello")) as events:
            await anext(events)
        assert runner.closed
        assert session.build_context() == [{"role": "user", "content": "Hello"}]
    run(check())
