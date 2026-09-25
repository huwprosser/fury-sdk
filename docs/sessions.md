# Optional session storage

`fury.sessions` provides append-only conversation storage. Importing it performs
no disk I/O. It uses only the standard library: no installation extra, database
driver, global session, or changes to existing `Agent` calls are required.

- `Agent` / `Runner` generate responses.
- `HistoryManager` bounds model context.
- `SessionManager` stores the durable log and reconstructs context.
- `SessionStorage` supplies persistence, independently of databases and ORMs.

## JSONL quick start

Inside your application's async function, with an existing `agent`:

```python
from fury.sessions import JSONLStorage, SessionManager

storage = JSONLStorage("./sessions")
session = await SessionManager.load_or_create(storage=storage, id="customer-123")

runner = agent.runner()  # A fresh runner per turn.
async for event in session.stream(runner, "Hello"):
    if event.content:
        print(event.content, end="", flush=True)
```

`stream()` is a small explicit recording adapter, not a persistent Agent subclass.
It records the input and each `history_delta.message` before forwarding the event.
Tool calls/results and multimodal messages are preserved. It does not store
reasoning tokens, UI events, or automatically persist every call to the agent.
A message dictionary can be passed instead of a string. Other keyword arguments
are forwarded to `runner.chat()`.

### Interruption and shutdown

`runner.interrupt()` retains unfinished assistant text; `runner.cancel()` discards
it. Messages already committed (including tool calls/results) remain in either
case. The adapter records only the uncommitted text, so narration from a previous
tool round is not duplicated. This bridges Fury's existing behavior where the
runner appends interrupted text to its input list instead of emitting a delta.

Consume the stream to exhaustion. If you may break early, close it explicitly:

```python
from contextlib import aclosing

runner = agent.runner()
async with aclosing(session.stream(runner, "Explain TCP")) as events:
    async for event in events:
        if should_stop():
            runner.interrupt()  # Omit this to discard unfinished text.
            break
```

Task cancellation, early closure, and provider failures discard uncommitted text
unless interruption was explicitly requested. Committed records are not rolled
back. The adapter does not repair unfinished tool-call/result pairs; applications
may need a recovery policy before resuming an interrupted tool round.

Storage operations finish before cancellation releases their lock. A cancelled
append may therefore have committed: do not blindly retry it. Abrupt process
termination cannot execute async cleanup. No token-by-token crash recovery is
promised.

### Manual recording

For full control, record messages yourself instead of using `stream()`:

```python
await session.append_message({"role": "user", "content": "Hello"})
async for event in agent.runner().chat(session.build_context()):
    if event.history_delta:
        await session.append_message(event.history_delta.message)
```

Do not combine manual recording with the adapter for the same turn. Manual
recording must handle interrupted partials itself; the runner does not emit those
as history deltas. See [interruption.md](interruption.md).

## Metadata and application data

```python
session = await SessionManager.create(
    storage=storage,
    metadata={"user_id": "123", "title": "Holiday planning"},
)
await session.update_metadata({"title": "Japan itinerary", "nullable": None})
await session.delete_metadata("title")
print(session.metadata)

await session.append_custom("checkout.completed", {"order_id": "456", "total": 29.99})
await session.append_custom("app.state", {"selected_trip": "japan"})
```

Metadata updates merge at the **top level**; nested objects are replaced, not
recursively merged. `None` is a value, not deletion. Metadata is a separate
namespace and cannot overwrite session identity/version. Custom entries are
ordered events or snapshots, not mutable attributes.

Neither metadata nor custom entries enter model context automatically. All values
must be JSON-compatible (string object keys; no NaN, Infinity, tuples, or arbitrary
Python objects). Store large assets externally and retain references. This module
does not encrypt stored conversations. Treat tool outputs and metadata as potentially
sensitive, and set directory/database permissions accordingly.

`entries`, `metadata`, and `build_context()` return copies: mutating them cannot
bypass persistence.

## Lifecycle and compaction

```python
session = SessionManager.in_memory(id="draft", metadata={"title": "Draft"})
await session.append_message({"role": "user", "content": "Hello"})
await session.promote(storage=storage)  # Preserve ID and existing entries.

restored = await SessionManager.open(storage=storage, id=session.id)
print(restored.is_persisted())

await restored.append_compaction("Summary of older turns", 12000, retained_tail=[])
context = restored.build_context()
```

`create()` rejects existing IDs. `open()` requires an existing session.
`load_or_create()` creates only when absent; it does not overwrite corruption.
Its `metadata` argument is used only on creation. IDs allow 1–128 ASCII letters,
digits, underscores, or hyphens; omitted IDs are generated UUIDs.

Context contains the latest compaction summary and retained tail, then subsequent
messages. Earlier entries remain an audit trail. The module does not decide when
to summarize or trim context; use [HistoryCompactor](history_compactor.md) and
[HistoryManager](history_manager.md) separately. Unknown entries, including
application-specific notifications, are preserved but excluded from context.

Promotion creates a header, then appends any in-memory entries. These are two
operations, not a cross-operation transaction: an append failure can leave a
header-only persisted session. Errors propagate and the failed manager rejects
further writes; callers must reopen and recover explicitly.

## Backend contract

```python
class SessionStorage(Protocol):
    async def create(self, header: dict) -> None: ...
    async def read(self, session_id: str) -> SessionRecord: ...
    async def append(self, session_id: str, entries: list[dict]) -> None: ...
```

`SessionRecord(header=..., entries=...)` holds a versioned header and ordered log.

- Duplicate creation raises `SessionExistsError`.
- Missing read/append raises `SessionNotFoundError`.
- Reads return consistent ordered snapshots; append batches are atomic during
  normal operation. Backend crash durability must be documented separately.
- Storage failures propagate; there is no silent in-memory fallback.
- **One active writer per session**, across instances and processes. Local asyncio
  locks do not provide distributed ownership or optimistic concurrency control.
- Applications own backend connections and close them themselves.
- Async callbacks are required; use `asyncio.to_thread()` for blocking clients.

The callback adapter accepts `create`, `read`, and `append` functions. Listing and
deletion are optional capabilities, not part of the minimum contract. JSONL
provides `await storage.list()` (headers) and `await storage.delete(id)`; database
applications can implement their own queries and authorization.

## MongoDB example

Install the optional driver **in your application**, not Fury:

```bash
pip install 'pymongo>=4.13'
```

This complete example stores one MongoDB document per session:

```python
import asyncio

from pymongo import AsyncMongoClient
from pymongo.errors import DuplicateKeyError
from fury.sessions import (
    CallbackStorage,
    SessionExistsError,
    SessionManager,
    SessionNotFoundError,
    SessionRecord,
)


async def main():
    client = AsyncMongoClient("mongodb://localhost:27017")
    collection = client.my_app.sessions

    async def create_session(header):
        try:
            await collection.insert_one({
                "_id": header["id"],
                "header": header,
                "entries": [],
            })
        except DuplicateKeyError as exc:
            raise SessionExistsError(header["id"]) from exc

    async def read_session(session_id):
        document = await collection.find_one({"_id": session_id})
        if document is None:
            raise SessionNotFoundError(session_id)
        return SessionRecord(
            header=document["header"],
            entries=document["entries"],
        )

    async def append_entries(session_id, entries):
        result = await collection.update_one(
            {"_id": session_id},
            {"$push": {"entries": {"$each": entries}}},
        )
        if result.matched_count == 0:
            raise SessionNotFoundError(session_id)

    storage = CallbackStorage(
        create=create_session,
        read=read_session,
        append=append_entries,
    )

    try:
        session = await SessionManager.load_or_create(
            storage=storage, id="customer-123",
        )
        await session.update_metadata({"title": "Japan itinerary"})
        await session.append_message({"role": "user", "content": "Hello"})
        await session.append_custom("app.state", {"selected_trip": "japan"})

        # This also works after restarting the application.
        restored = await SessionManager.open(storage=storage, id=session.id)
        print(restored.metadata)
        print(restored.build_context())

        # With an existing Fury agent:
        # async for event in restored.stream(agent.runner(), "Plan my trip"):
        #     if event.content:
        #         print(event.content, end="", flush=True)
    finally:
        await client.close()


asyncio.run(main())
```

MongoDB's unique `_id` enforces unique creation; `$push/$each` atomically appends
an ordered batch within one document. Use acknowledged writes. Metadata updates
remain log entries, reconstructed by SessionManager rather than duplicated into
MongoDB-specific fields.

**Limits:** MongoDB documents have a 16 MiB limit. Long-lived sessions should use
separate header/entry collections, explicit per-session ordering, and transactions
for atomic batches. MongoDB also has BSON-specific key/value constraints (for
example integer range), which applications must account for. A network failure
can make a write's outcome uncertain; this contract does not promise exactly-once
retries. Use appropriate write concern and deployment settings for your durability
requirements. The single-writer rule still applies.

## JSONL format and recovery

Files are `directory/<timestamp>_<id>.jsonl`. Version 1 is compatible with the
voice-agent session format; existing files without metadata can be loaded.

```jsonl
{"type":"session","version":1,"id":"abc","timestamp":"2026-01-01T00:00:00Z","cwd":"/app","metadata":{}}
{"type":"message","timestamp":"...","message":{"role":"user","content":"Hello"}}
{"type":"metadata","timestamp":"...","set":{"title":"Trip"},"delete":[]}
{"type":"custom","timestamp":"...","customType":"app.state","data":{"step":1}}
{"type":"compaction","timestamp":"...","summary":"Earlier conversation","tokensBefore":1000,"retainedTail":[]}
```

Malformed unterminated final entries are ignored on read and truncated before the
next append. Malformed complete lines, invalid headers, unsupported versions, and
multiple files for an ID raise errors. Successful writes are flushed to the OS,
not fsynced. This backend is not power-loss transactional; a crash can leave a
partial record or partially written batch. Ordinary write errors attempt rollback.
