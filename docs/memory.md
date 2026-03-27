# Memory

Fury's durable memory layer is separate from chat history. Use it when you want facts or project context to survive across turns, sessions, or process restarts.

The memory tool is not exhuastive, and is intentionally focused on short, high quality memories. 
It does not use RAG, instead it takes the Anthropic-approach to memory, giving the Agent tools to search, write and edit memories to a datastore.
The agent is encouraged to kee memroies concise and keep track of our memory consumption (default is 2500 characters).

`MEMORY.md` has been deprecated.

## Model

Memory is organized by explicit named scopes. A scope is just a string such as `my-project` or `customer-a`.

That means:

- two agents can share a process and still stay memory-isolated
- the model only sees the scope you bind it to

## Quick Start

The simplest setup is to let `Agent` manage prompt injection and the memory tool for you:

```python
from fury import Agent

agent = Agent(
    model="your-model-name",
    system_prompt="You are a helpful assistant.",
    memory_scope="my-project",
)
```

That does three things:

1. creates a durable `MemoryStore` at `.fury/memory` if you did not pass one
2. injects the latest memory snapshot for `my-project` into the system prompt on each turn
3. registers a `memory` tool bound only to `my-project`

If the agent writes memory during a turn, later turns will automatically see the updated snapshot.

## Shared Store, Multiple Airgapped Agents

```python
from fury import Agent, MemoryStore

store = MemoryStore(".fury/memory")

alpha = Agent(
    model="your-model-name",
    system_prompt="You help project alpha.",
    memory_store=store,
    memory_scope="project-alpha",
)

beta = Agent(
    model="your-model-name",
    system_prompt="You help project beta.",
    memory_store=store,
    memory_scope="project-beta",
)
```

These agents share the same backing directory but cannot see each other's memory unless you explicitly bind them to the same scope string.

Each turn an agent processes, the system prompt will be reconstructed with the memory block. If something has changed in the memory, this will be automatically added to the system prompt on the next turn. [Room for improvement]

## Low-Level API

### `MemoryStore`

```python
from fury import MemoryStore

store = MemoryStore(
    root_dir=".fury/memory",
    char_limit=2500,
)
```

Methods:

- `add(scope, content, source="agent", pinned=False)`
- `replace(scope, content, entry_id=None, match_text=None, pinned=None)`
- `remove(scope, entry_id=None, match_text=None)`
- `get_scope(scope)`
- `get_state(scope)`
- `capture_snapshot(scope)`

### `create_memory_tool`

```python
from fury import create_memory_tool

memory_tool = create_memory_tool(store, "my-project")
```

The returned tool is airgapped to that scope. Its actions are:

- `add`
- `replace`
- `remove`
- `list`

There is no runtime parameter for choosing another scope.

## Prompt Helpers

If you want to manage the prompt manually:

```python
from fury import compose_prompt_with_memory

snapshot = store.capture_snapshot("my-project")
prompt = compose_prompt_with_memory("Base prompt", snapshot)
```

## Safety

The store rejects memory content that matches a small set of prompt-injection and exfiltration patterns, and it blocks invisible Unicode control characters commonly used to hide malicious text.

## Relationship To History

Use `HistoryManager` or `StaticHistoryManager` for short-term conversational context. Use named memory scopes for durable context across sessions.
