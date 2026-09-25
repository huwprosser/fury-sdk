"""Opt-in, backend-independent append-only sessions. No I/O occurs on import."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import uuid
from contextlib import aclosing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

VERSION = 1
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


class SessionNotFoundError(FileNotFoundError):
    pass


class SessionExistsError(FileExistsError):
    pass


class InvalidSessionError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _validate_id(value: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError("Session id must contain 1–128 letters, digits, underscores or hyphens")
    return value


def _snapshot(value):
    # Reject non-JSON values, including NaN/Infinity, without converting dict keys.
    def check(item):
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise TypeError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif item is not None and not isinstance(item, (str, bool, int, float)):
            raise TypeError("Session data must be JSON-compatible")
    check(value)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


@dataclass
class SessionRecord:
    header: dict[str, Any]
    entries: list[dict[str, Any]]


class SessionStorage(Protocol):
    """Single writer per session; append batches must be atomic and ordered.

    Raise SessionExistsError on duplicate create, SessionNotFoundError on missing
    read/append. Errors must propagate. Implementations own their connections.
    """

    async def create(self, header: dict[str, Any]) -> None: ...
    async def read(self, session_id: str) -> SessionRecord: ...
    async def append(self, session_id: str, entries: list[dict[str, Any]]) -> None: ...


class CallbackStorage:
    """Adapt three async application callbacks to SessionStorage."""

    def __init__(self, *, create: Callable[..., Awaitable[None]],
                 read: Callable[..., Awaitable[SessionRecord]],
                 append: Callable[..., Awaitable[None]]):
        self.create, self.read, self.append = create, read, append


class JSONLStorage:
    """Timestamp/ID JSONL files. One active writer per session, across all processes.

    Successful writes are flushed to the OS, not fsynced. Process/power failure
    can leave a partial last line. This is not a transactional database.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser()
        self._lock = asyncio.Lock()

    def _find(self, session_id):
        _validate_id(session_id)
        paths = sorted(self.directory.glob(f"*_{session_id}.jsonl"))
        if not paths:
            raise SessionNotFoundError(session_id)
        if len(paths) != 1:
            raise InvalidSessionError(f"Multiple files for session {session_id}")
        return paths[0]

    def _read(self, session_id):
        path = self._find(session_id)
        raw = path.read_bytes()
        lines = raw.splitlines(keepends=True)
        if not lines:
            raise InvalidSessionError(f"Empty session: {path}")
        entries, valid_end = [], 0
        for index, line in enumerate(lines):
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise InvalidSessionError(f"Expected object on line {index + 1}")
            except (ValueError, UnicodeDecodeError) as exc:
                # Only an unterminated, malformed last entry is recoverable.
                if index > 0 and index == len(lines) - 1 and not line.endswith(b"\n"):
                    break
                raise InvalidSessionError(f"Invalid line {index + 1} of {path}") from exc
            if index == 0:
                header = value
            else:
                entries.append(value)
            valid_end += len(line)
        _validate_record(SessionRecord(header, entries), session_id)
        return path, SessionRecord(header, entries), valid_end, raw

    async def create(self, header):
        header = _snapshot(header)
        _validate_record(SessionRecord(header, []), header.get("id"))
        async with self._lock:
            await _finish_io(asyncio.to_thread(self._create, header))

    def _create(self, header):
        session_id = header["id"]
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            self._find(session_id)
        except SessionNotFoundError:
            pass
        else:
            raise SessionExistsError(session_id)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3] + "Z"
        path = self.directory / f"{stamp}_{session_id}.jsonl"
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(header, ensure_ascii=False) + "\n")
        except FileExistsError as exc:
            raise SessionExistsError(session_id) from exc

    async def read(self, session_id):
        async with self._lock:
            _, record, _, _ = await _finish_io(asyncio.to_thread(self._read, session_id))
            return record

    async def append(self, session_id, entries):
        entries = _snapshot(entries)
        async with self._lock:
            await _finish_io(asyncio.to_thread(self._append, session_id, entries))

    def _append(self, session_id, entries):
        path, _, end, raw = self._read(session_id)
        prefix = b"\n" if end and raw[end - 1:end] != b"\n" else b""
        payload = prefix + b"".join(
            (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8") for entry in entries
        )
        with path.open("r+b") as handle:
            handle.truncate(end)  # Repair a torn final append before writing again.
            handle.seek(end)
            try:
                handle.write(payload)
                handle.flush()
            except BaseException:
                handle.seek(end)
                handle.truncate()
                raise

    async def list(self) -> list[dict[str, Any]]:
        """Optional capability: return headers; no application-specific previews."""
        def scan():
            headers = []
            for path in sorted(self.directory.glob("*.jsonl")):
                with path.open(encoding="utf-8") as handle:
                    header = json.loads(handle.readline())
                _validate_record(SessionRecord(header, []), header.get("id"))
                headers.append(header)
            return headers
        async with self._lock:
            return await _finish_io(asyncio.to_thread(scan))

    async def delete(self, session_id):
        async with self._lock:
            await _finish_io(asyncio.to_thread(lambda: self._find(session_id).unlink()))


async def _finish_io(operation):
    """Do not release an operation's lock while cancelled background I/O runs."""
    task = asyncio.ensure_future(operation)
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return result


def _validate_record(record, session_id):
    _validate_id(session_id)
    header = record.header
    if (header.get("type") != "session" or header.get("version") != VERSION
            or header.get("id") != session_id):
        raise InvalidSessionError("Invalid session header, id or unsupported version")
    if not isinstance(header.get("timestamp"), str):
        raise InvalidSessionError("Missing session timestamp")
    if not isinstance(header.get("metadata", {}), dict):
        raise InvalidSessionError("Session metadata must be an object")
    if not isinstance(record.entries, list) or not all(isinstance(e, dict) for e in record.entries):
        raise InvalidSessionError("Session entries must be objects")
    for entry in record.entries:
        if entry.get("type") == "metadata":
            if (not isinstance(entry.get("set", {}), dict)
                    or not isinstance(entry.get("delete", []), list)
                    or not all(isinstance(key, str) for key in entry.get("delete", []))):
                raise InvalidSessionError("Invalid metadata update")
    _snapshot(header)
    _snapshot(record.entries)


class SessionManager:
    """One conversation log. Storage is explicit; metadata never becomes context."""

    def __init__(self, record: SessionRecord, storage: SessionStorage | None = None):
        _validate_record(record, record.header.get("id"))
        self._header = _snapshot(record.header)
        self._entries = _snapshot(record.entries)
        self._storage = storage
        self._promotion_failed = False
        self._lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()

    @classmethod
    def in_memory(cls, *, id=None, metadata=None):
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary")
        header = {"type": "session", "version": VERSION,
                  "id": _validate_id(id if id is not None else uuid.uuid4().hex),
                  "timestamp": _now(), "cwd": os.getcwd(), "metadata": metadata or {}}
        return cls(SessionRecord(header, []))

    @classmethod
    async def create(cls, *, storage, id=None, metadata=None):
        session = cls.in_memory(id=id, metadata=metadata)
        await session.promote(storage=storage)
        return session

    @classmethod
    async def open(cls, *, storage, id):
        record = await storage.read(_validate_id(id))
        _validate_record(record, id)
        return cls(record, storage)

    @classmethod
    async def load_or_create(cls, *, storage, id, metadata=None):
        try:
            return await cls.open(storage=storage, id=id)
        except SessionNotFoundError:
            try:
                return await cls.create(storage=storage, id=id, metadata=metadata)
            except SessionExistsError:
                return await cls.open(storage=storage, id=id)

    async def promote(self, *, storage):
        async with self._lock:
            if self._promotion_failed:
                raise RuntimeError("Promotion failed; reopen and recover the persisted session")
            if self._storage is not None:
                if self._storage is not storage:
                    raise ValueError("Session is already attached to another storage")
                return
            async def commit():
                await storage.create(_snapshot(self._header))
                # Attach immediately: if copying entries fails, report failure rather
                # than pretending the ID does not already exist on disk.
                self._storage = storage
                if self._entries:
                    try:
                        await storage.append(self.id, _snapshot(self._entries))
                    except BaseException:
                        self._promotion_failed = True
                        raise
            await _finish_io(commit())

    async def _append(self, entry):
        entry = _snapshot(entry)
        async with self._lock:
            if self._promotion_failed:
                raise RuntimeError("Promotion failed; reopen and recover the persisted session")
            async def commit():
                if self._storage is not None:
                    await self._storage.append(self.id, [entry])
                self._entries.append(entry)
            await _finish_io(commit())

    async def append_message(self, message):
        if not isinstance(message, dict):
            raise TypeError("message must be a dictionary")
        await self._append({"type": "message", "timestamp": _now(), "message": message})

    async def append_custom(self, custom_type, data=None):
        if not isinstance(custom_type, str):
            raise TypeError("custom_type must be a string")
        await self._append({"type": "custom", "timestamp": _now(),
                            "customType": custom_type, "data": data})

    async def append_compaction(self, summary, tokens_before, retained_tail=None):
        entry = {"type": "compaction", "timestamp": _now(), "summary": summary,
                 "tokensBefore": tokens_before}
        if retained_tail:
            entry["retainedTail"] = retained_tail
        await self._append(entry)

    async def update_metadata(self, values):
        if not isinstance(values, dict):
            raise TypeError("metadata must be a dictionary")
        await self._append({"type": "metadata", "timestamp": _now(), "set": values, "delete": []})

    async def delete_metadata(self, *keys):
        if not all(isinstance(key, str) for key in keys):
            raise TypeError("metadata keys must be strings")
        await self._append({"type": "metadata", "timestamp": _now(), "set": {}, "delete": list(keys)})

    @property
    def id(self):
        return self._header["id"]

    @property
    def entries(self):
        return copy.deepcopy(self._entries)

    @property
    def metadata(self):
        result = copy.deepcopy(self._header.get("metadata", {}))
        for entry in self._entries:
            if entry.get("type") == "metadata":
                result.update(copy.deepcopy(entry.get("set", {})))
                for key in entry.get("delete", []):
                    result.pop(key, None)
        return result

    def is_persisted(self):
        return self._storage is not None

    def build_context(self):
        context, start = [], 0
        for index in range(len(self._entries) - 1, -1, -1):
            entry = self._entries[index]
            if entry.get("type") == "compaction":
                if entry.get("summary"):
                    context.append({"role": "user", "content":
                                    f"[Previous conversation summary]\n{entry['summary']}"})
                context.extend(entry.get("retainedTail") or [])
                start = index + 1
                break
        context.extend(entry["message"] for entry in self._entries[start:]
                       if entry.get("type") == "message" and isinstance(entry.get("message"), dict))
        return copy.deepcopy(context)

    async def stream(self, runner, user_input, **options):
        """Record a turn, yielding Fury events unchanged. Use a fresh Runner per turn.

        Consume to exhaustion or use contextlib.aclosing. Explicit interrupt keeps
        unfinished text; cancel, task cancellation, and early close discard it.
        Already-recorded messages/tool results are never rolled back.
        """
        async with self._turn_lock:
            message = {"role": "user", "content": user_input} if isinstance(user_input, str) else user_input
            await self.append_message(message)
            committed_text = ""
            recording_failed = False
            async with aclosing(runner.chat(self.build_context(), **options)) as events:
                try:
                    async for event in events:
                        if event.history_delta:
                            delta = event.history_delta.message
                            try:
                                await self.append_message(delta)
                            except BaseException:
                                # A cancelled write may already have committed. Do
                                # not retry its text from the interruption buffer.
                                recording_failed = True
                                raise
                            if delta.get("role") == "assistant" and isinstance(delta.get("content"), str):
                                committed_text += delta["content"]
                        yield event
                finally:
                    # Runtime finalization mutates its history instead of emitting a
                    # delta for interruption. Record only text not committed already.
                    if runner.interrupted and not recording_failed:
                        partial = runner.partial_response
                        if partial.startswith(committed_text):
                            partial = partial[len(committed_text):]
                        if partial:
                            await self.append_message({"role": "assistant", "content": partial})
