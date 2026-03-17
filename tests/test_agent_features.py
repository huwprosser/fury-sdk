import asyncio
import json
import sys
import time
import types

import httpx
import numpy as np
import pytest
from conftest import (
    FakeCompletion,
    FakeDelta,
    FakeToolCallChunk,
    SequencedCreate,
    SlowFakeCompletion,
    make_fake_client,
)

from fury import Agent, create_tool


def collect_chat(agent, history, **kwargs):
    async def run():
        events = []
        async for event in agent.chat(history, **kwargs):
            events.append(event)
        return events

    return asyncio.run(run())


def test_agent_completes_basic_text_conversation():
    create = SequencedCreate(
        [FakeCompletion([FakeDelta(content="Hello"), FakeDelta(content=" world")])]
    )
    agent = Agent(model="test-model", system_prompt="You are helpful.")
    agent.client = make_fake_client(create)

    response = agent.ask("Say hello.", history=[])

    assert response == "Hello world"


def test_agent_preserves_system_prompt_when_calling_with_external_history():
    create = SequencedCreate(
        [
            FakeCompletion([FakeDelta(content="ok")]),
            FakeCompletion([FakeDelta(content="ok")]),
        ]
    )
    agent = Agent(model="test-model", system_prompt="System prompt")
    agent.client = make_fake_client(create)

    collect_chat(agent, [{"role": "user", "content": "hello"}])
    collect_chat(
        agent,
        [
            {"role": "system", "content": "Custom system"},
            {"role": "user", "content": "hello"},
        ],
    )

    first_messages = create.calls[0]["messages"]
    second_messages = create.calls[1]["messages"]
    assert first_messages[0] == {"role": "system", "content": "System prompt"}
    assert sum(1 for message in first_messages if message["role"] == "system") == 1
    assert second_messages[0] == {"role": "system", "content": "Custom system"}
    assert sum(1 for message in second_messages if message["role"] == "system") == 1


def test_agent_streams_plain_text_in_order():
    create = SequencedCreate(
        [
            FakeCompletion(
                [FakeDelta(content="A"), FakeDelta(content="B"), FakeDelta(content="C")]
            )
        ]
    )
    agent = Agent(model="test-model", system_prompt="You are helpful.")
    agent.client = make_fake_client(create)

    events = collect_chat(
        agent,
        [{"role": "user", "content": "stream text"}],
        reasoning=False,
    )

    assert "".join(event.content for event in events if event.content) == "ABC"


def test_agent_streams_reasoning_only_when_enabled():
    async def create(**kwargs):
        if "extra_body" in kwargs:
            return FakeCompletion([FakeDelta(content="plain")])
        return FakeCompletion(
            [FakeDelta(reasoning_content="thinking"), FakeDelta(content="answer")]
        )

    agent = Agent(model="test-model", system_prompt="You are helpful.")
    agent.client = make_fake_client(create)

    disabled = collect_chat(
        agent,
        [{"role": "user", "content": "no reasoning"}],
        reasoning=False,
    )
    enabled = collect_chat(
        agent,
        [{"role": "user", "content": "with reasoning"}],
        reasoning=True,
    )

    assert [event.reasoning for event in disabled if event.reasoning] == []
    assert [event.reasoning for event in enabled if event.reasoning] == ["thinking"]
    assert "".join(event.content for event in enabled if event.content) == "answer"


def test_agent_prunes_unfinished_streamed_sentences_when_requested():
    create = SequencedCreate(
        [
            FakeCompletion(
                [
                    FakeDelta(content="Hello there. This is"),
                    FakeDelta(content=" a test!"),
                    FakeDelta(content=" And another sentence."),
                    FakeDelta(content=" trailing"),
                ]
            )
        ]
    )
    agent = Agent(model="test-model", system_prompt="You are helpful.")
    agent.client = make_fake_client(create)

    events = collect_chat(
        agent,
        [{"role": "user", "content": "prune"}],
        prune_unfinished_sentences=True,
    )

    assert [event.content for event in events if event.content] == [
        "Hello there. This is a test!",
        " And another sentence.",
    ]


def test_create_tool_uses_input_and_output_schemas():
    tool = create_tool(
        "noop",
        "Do nothing.",
        lambda: None,
        {"type": "object", "properties": {}, "required": []},
        {"type": "object", "properties": {}, "required": []},
    )

    assert tool.input_schema == {"type": "object", "properties": {}, "required": []}
    assert tool.output_schema == {"type": "object", "properties": {}, "required": []}


def test_agent_executes_single_tool_and_returns_final_answer():
    called = {}

    def add(a, b):
        called["args"] = (a, b)
        return {"result": a + b}

    tool = create_tool(
        "add",
        "Add two numbers.",
        add,
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        {"type": "object", "properties": {"result": {"type": "integer"}}},
    )

    create = SequencedCreate(
        [
            FakeCompletion(
                [
                    FakeDelta(
                        tool_calls=[
                            FakeToolCallChunk(
                                0,
                                id="call_1",
                                name="add",
                                arguments='{"a": 2, "b": 3}',
                            )
                        ]
                    )
                ]
            ),
            FakeCompletion([FakeDelta(content="The sum is 5.")]),
        ]
    )
    agent = Agent(model="test-model", system_prompt="You are helpful.", tools=[tool])
    agent.client = make_fake_client(create)

    events = collect_chat(agent, [{"role": "user", "content": "add numbers"}])

    assert called["args"] == (2, 3)
    assert any(
        event.tool_call and event.tool_call.arguments == {"a": 2, "b": 3}
        for event in events
    )
    assert any(
        event.tool_call and event.tool_call.result == {"result": 5} for event in events
    )
    assert (
        "".join(event.content for event in events if event.content) == "The sum is 5."
    )


def test_agent_streams_tool_ui_events_for_sync_tools_without_exposing_emit_in_schema():
    def search(query, emit):
        assert callable(emit)
        emit(
            {
                "id": "search-1",
                "title": "Searching for latest react version",
                "type": "tool_call",
            }
        )
        time.sleep(0.01)
        emit(
            {
                "id": "search-1",
                "title": "Found latest react version",
                "type": "tool_call",
            }
        )
        return {"status": "done", "query": query}

    tool = create_tool(
        id="search",
        description="Search for something.",
        execute=search,
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "query": {"type": "string"},
            },
        },
    )

    create = SequencedCreate(
        [
            FakeCompletion(
                [
                    FakeDelta(
                        tool_calls=[
                            FakeToolCallChunk(
                                0,
                                id="call_1",
                                name="search",
                                arguments='{"query": "latest react version"}',
                            )
                        ]
                    )
                ]
            ),
            FakeCompletion([FakeDelta(content="Search complete.")]),
        ]
    )
    agent = Agent(model="test-model", system_prompt="You are helpful.", tools=[tool])
    agent.client = make_fake_client(create)

    events = collect_chat(agent, [{"role": "user", "content": "search"}])

    ui_events = [event.tool_ui for event in events if event.tool_ui]
    assert [(event.id, event.title, event.type) for event in ui_events] == [
        ("search-1", "Searching for latest react version", "tool_call"),
        ("search-1", "Found latest react version", "tool_call"),
    ]
    assert [
        "tool_call"
        if event.tool_call and event.tool_call.arguments is not None
        else "tool_ui"
        if event.tool_ui
        else "result"
        if event.tool_call and event.tool_call.arguments is None
        else "content"
        for event in events
    ] == ["tool_call", "tool_ui", "tool_ui", "result", "content"]

    search_schema = next(
        tool_def["function"]["parameters"]
        for tool_def in create.calls[0]["tools"]
        if tool_def["function"]["name"] == "search"
    )
    assert "emit" not in search_schema.get("properties", {})


def test_agent_streams_tool_ui_events_for_async_tools():
    async def search(query, emit):
        emit({"id": "phase-1", "title": f"Queued {query}", "type": "other"})
        await asyncio.sleep(0)
        emit({"id": "phase-2", "title": f"Fetched {query}", "type": "tool_call"})
        return {"status": "done"}

    tool = create_tool(
        id="search_async",
        description="Search asynchronously.",
        execute=search,
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        output_schema={"type": "object", "properties": {"status": {"type": "string"}}},
    )

    create = SequencedCreate(
        [
            FakeCompletion(
                [
                    FakeDelta(
                        tool_calls=[
                            FakeToolCallChunk(
                                0,
                                id="call_1",
                                name="search_async",
                                arguments='{"query": "release notes"}',
                            )
                        ]
                    )
                ]
            ),
            FakeCompletion([FakeDelta(content="Async search complete.")]),
        ]
    )
    agent = Agent(model="test-model", system_prompt="You are helpful.", tools=[tool])
    agent.client = make_fake_client(create)

    events = collect_chat(agent, [{"role": "user", "content": "search"}])

    assert [
        (event.tool_ui.id, event.tool_ui.title, event.tool_ui.type)
        for event in events
        if event.tool_ui
    ] == [
        ("phase-1", "Queued release notes", "other"),
        ("phase-2", "Fetched release notes", "tool_call"),
    ]
    assert any(
        event.tool_call and event.tool_call.result == {"status": "done"}
        for event in events
    )


def test_agent_filters_hallucinated_tool_arguments():
    observed = {}

    def echo(text):
        observed["text"] = text
        return {"text": text}

    tool = create_tool(
        "echo",
        "Echo text.",
        echo,
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        {"type": "object", "properties": {"text": {"type": "string"}}},
    )

    create = SequencedCreate(
        [
            FakeCompletion(
                [
                    FakeDelta(
                        tool_calls=[
                            FakeToolCallChunk(
                                0,
                                id="call_1",
                                name="echo",
                                arguments='{"text": "hi", "ignored": "nope"}',
                            )
                        ]
                    )
                ]
            ),
            FakeCompletion([FakeDelta(content="done")]),
        ]
    )
    agent = Agent(model="test-model", system_prompt="You are helpful.", tools=[tool])
    agent.client = make_fake_client(create)

    events = collect_chat(agent, [{"role": "user", "content": "echo hi"}])

    assert observed == {"text": "hi"}
    assert any(
        event.tool_call and event.tool_call.arguments == {"text": "hi"}
        for event in events
    )


def test_agent_surfaces_tool_execution_errors_without_crashing_conversation():
    def explode():
        raise RuntimeError("boom")

    tool = create_tool(
        "explode",
        "Fail on purpose.",
        explode,
        {"type": "object", "properties": {}, "required": []},
        {"type": "object", "properties": {}},
    )

    def second_response(kwargs):
        tool_messages = [m for m in kwargs["messages"] if m.get("role") == "tool"]
        assert tool_messages[-1]["content"] == "Error: boom"
        return FakeCompletion([FakeDelta(content="Recovered after tool failure.")])

    create = SequencedCreate(
        [
            FakeCompletion(
                [
                    FakeDelta(
                        tool_calls=[
                            FakeToolCallChunk(
                                0, id="call_1", name="explode", arguments="{}"
                            )
                        ]
                    )
                ]
            ),
            second_response,
        ]
    )
    agent = Agent(model="test-model", system_prompt="You are helpful.", tools=[tool])
    agent.client = make_fake_client(create)

    events = collect_chat(agent, [{"role": "user", "content": "explode"}])

    assert any(
        event.tool_call and event.tool_call.result == "Error: boom" for event in events
    )
    assert "".join(event.content for event in events if event.content) == (
        "Recovered after tool failure."
    )


def test_agent_handles_invalid_tool_call_json_gracefully():
    create = SequencedCreate(
        [
            FakeCompletion(
                [
                    FakeDelta(
                        tool_calls=[
                            FakeToolCallChunk(
                                0,
                                id="call_1",
                                name="noop",
                                arguments='{"broken": ',
                            )
                        ]
                    )
                ]
            ),
            FakeCompletion([FakeDelta(content="Recovered")]),
        ]
    )
    agent = Agent(model="test-model", system_prompt="You are helpful.")
    agent.available_functions["noop"] = lambda: None
    agent.client = make_fake_client(create)

    events = collect_chat(agent, [{"role": "user", "content": "bad json"}])

    error_messages = [event.content for event in events if event.content]
    assert "Error decoding arguments for noop" in error_messages[0]
    assert error_messages[-1] == "Recovered"


def test_agent_executes_parallel_tool_wrapper_for_independent_tools():
    def one():
        return 1

    def two():
        return 2

    create = SequencedCreate(
        [
            FakeCompletion(
                [
                    FakeDelta(
                        tool_calls=[
                            FakeToolCallChunk(
                                0,
                                id="call_1",
                                name="multi_tool_use.parallel",
                                arguments=json.dumps(
                                    {
                                        "tool_uses": [
                                            {
                                                "recipient_name": "one",
                                                "parameters": {},
                                            },
                                            {
                                                "recipient_name": "two",
                                                "parameters": {},
                                            },
                                        ]
                                    }
                                ),
                            )
                        ]
                    )
                ]
            ),
            FakeCompletion([FakeDelta(content="Parallel complete")]),
        ]
    )
    agent = Agent(model="test-model", system_prompt="You are helpful.")
    agent.available_functions["one"] = one
    agent.available_functions["two"] = two
    agent.client = make_fake_client(create)

    events = collect_chat(agent, [{"role": "user", "content": "run both"}])

    result_events = [
        event.tool_call.result
        for event in events
        if event.tool_call and event.tool_call.result is not None
    ]
    assert result_events == [
        [
            {"recipient_name": "one", "result": 1},
            {"recipient_name": "two", "result": 2},
        ]
    ]


def test_chat_cancel_discards_partial_response_and_closes_stream():
    completion = SlowFakeCompletion(
        [FakeDelta(content="Hello"), FakeDelta(content=" world")]
    )
    create = SequencedCreate([completion])
    agent = Agent(model="test-model", system_prompt="You are helpful.")
    agent.client = make_fake_client(create)

    async def run():
        history = [{"role": "user", "content": "stream text"}]
        runner = agent.runner()
        events = []

        async for event in runner.chat(history):
            events.append(event)
            if event.content == "Hello":
                runner.cancel()

        return history, events, runner

    history, events, runner = asyncio.run(run())

    assert [event.content for event in events if event.content] == ["Hello"]
    assert history == [{"role": "user", "content": "stream text"}]
    assert runner.cancelled is True
    assert runner.partial_response == "Hello"
    assert completion.closed is True


def test_chat_interrupt_preserves_partial_response_in_history():
    completion = SlowFakeCompletion(
        [FakeDelta(content="Hello"), FakeDelta(content=" world")]
    )
    create = SequencedCreate([completion])
    agent = Agent(model="test-model", system_prompt="You are helpful.")
    agent.client = make_fake_client(create)

    async def run():
        history = [{"role": "user", "content": "stream text"}]
        runner = agent.runner()
        events = []

        async for event in runner.chat(history):
            events.append(event)
            if event.content == "Hello":
                runner.interrupt()

        return history, events, runner

    history, events, runner = asyncio.run(run())

    assert [event.content for event in events if event.content] == ["Hello"]
    assert history == [
        {"role": "user", "content": "stream text"},
        {"role": "assistant", "content": "Hello"},
    ]
    assert runner.interrupted is True
    assert runner.partial_response == "Hello"
    assert completion.closed is True


def test_chat_interrupt_swallows_expected_transport_error(caplog):
    class InterruptReadErrorCompletion(SlowFakeCompletion):
        async def __anext__(self):
            if self.closed:
                raise httpx.ReadError("stream closed")
            if self._index > 0:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self.delay)
                except asyncio.TimeoutError:
                    pass
                self._wake.clear()
                if self.closed:
                    raise httpx.ReadError("stream closed")
            return await super(SlowFakeCompletion, self).__anext__()

    completion = InterruptReadErrorCompletion(
        [FakeDelta(content="Hello"), FakeDelta(content=" world")]
    )
    create = SequencedCreate([completion])
    agent = Agent(model="test-model", system_prompt="You are helpful.")
    agent.client = make_fake_client(create)

    async def run():
        history = [{"role": "user", "content": "stream text"}]
        runner = agent.runner()
        events = []

        async def interrupt_when_streaming():
            while runner.partial_response != "Hello":
                await asyncio.sleep(0.01)
            runner.interrupt()

        interrupter = asyncio.create_task(interrupt_when_streaming())
        async for event in runner.chat(history):
            events.append(event)
        await interrupter
        return history, events, runner

    with caplog.at_level("ERROR"):
        history, events, runner = asyncio.run(run())

    assert [event.content for event in events if event.content] == ["Hello"]
    assert history == [
        {"role": "user", "content": "stream text"},
        {"role": "assistant", "content": "Hello"},
    ]
    assert runner.interrupted is True
    assert "Error in chat" not in caplog.text


def test_runner_interrupt_preserves_partial_response_without_duplicate_history():
    completion = SlowFakeCompletion(
        [FakeDelta(content="Hello"), FakeDelta(content=" world")]
    )
    create = SequencedCreate([completion])
    agent = Agent(model="test-model", system_prompt="You are helpful.")
    agent.client = make_fake_client(create)

    async def run():
        history = [{"role": "user", "content": "stream text"}]
        runner = agent.runner()
        response = []

        async def interrupt_when_streaming():
            while runner.partial_response != "Hello":
                await asyncio.sleep(0.01)
            runner.interrupt()

        interrupter = asyncio.create_task(interrupt_when_streaming())
        async for event in runner.chat(history):
            if event.content:
                response.append(event.content)
        await interrupter
        return history, "".join(response), runner

    history, response, runner = asyncio.run(run())

    assert response == "Hello"
    assert history == [
        {"role": "user", "content": "stream text"},
        {"role": "assistant", "content": "Hello"},
    ]
    assert runner.interrupted is True


def test_runner_cancel_discards_partial_history():
    completion = SlowFakeCompletion(
        [FakeDelta(content="Hello"), FakeDelta(content=" world")]
    )
    create = SequencedCreate([completion])
    agent = Agent(model="test-model", system_prompt="You are helpful.")
    agent.client = make_fake_client(create)

    async def run():
        history = [{"role": "user", "content": "stream text"}]
        runner = agent.runner()
        response = []

        async def cancel_when_streaming():
            while runner.partial_response != "Hello":
                await asyncio.sleep(0.01)
            runner.cancel()

        canceller = asyncio.create_task(cancel_when_streaming())
        async for event in runner.chat(history):
            if event.content:
                response.append(event.content)
        await canceller
        return history, "".join(response), runner

    history, response, runner = asyncio.run(run())

    assert response == "Hello"
    assert history == [{"role": "user", "content": "stream text"}]
    assert runner.cancelled is True


def test_parallel_tool_wrapper_forwards_tool_ui_events_from_nested_tools():
    def one(emit):
        emit({"id": "one", "title": "Running one", "type": "tool_call"})
        return 1

    create = SequencedCreate(
        [
            FakeCompletion(
                [
                    FakeDelta(
                        tool_calls=[
                            FakeToolCallChunk(
                                0,
                                id="call_1",
                                name="multi_tool_use.parallel",
                                arguments=json.dumps(
                                    {
                                        "tool_uses": [
                                            {
                                                "recipient_name": "one",
                                                "parameters": {},
                                            }
                                        ]
                                    }
                                ),
                            )
                        ]
                    )
                ]
            ),
            FakeCompletion([FakeDelta(content="Parallel complete")]),
        ]
    )
    agent = Agent(model="test-model", system_prompt="You are helpful.")
    agent.available_functions["one"] = one
    agent.client = make_fake_client(create)

    events = collect_chat(agent, [{"role": "user", "content": "run one"}])

    assert [
        (event.tool_ui.id, event.tool_ui.title, event.tool_ui.type)
        for event in events
        if event.tool_ui
    ] == [("one", "Running one", "tool_call")]


def test_agent_rejects_nested_parallel_tool_calls():
    agent = Agent(model="test-model", system_prompt="You are helpful.")

    result = asyncio.run(
        agent._execute_parallel_tool(
            [
                {
                    "recipient_name": "multi_tool_use.parallel",
                    "parameters": {"tool_uses": []},
                }
            ]
        )
    )

    assert result == [
        {
            "recipient_name": "multi_tool_use.parallel",
            "error": "multi_tool_use.parallel cannot invoke itself",
        }
    ]


def test_agent_accepts_image_result_from_tool_and_appends_multimodal_followup_message():
    tool = create_tool(
        "camera",
        "Capture an image.",
        lambda: {"description": "snapshot", "image_base64": "ZmFrZQ=="},
        {"type": "object", "properties": {}, "required": []},
        {"type": "object", "properties": {}},
    )

    def second_response(kwargs):
        messages = kwargs["messages"]
        assert messages[-2]["role"] == "tool"
        assert messages[-2]["content"] == "snapshot"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"][1]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,ZmFrZQ=="
        )
        return FakeCompletion([FakeDelta(content="I can see the image.")])

    create = SequencedCreate(
        [
            FakeCompletion(
                [
                    FakeDelta(
                        tool_calls=[
                            FakeToolCallChunk(
                                0, id="call_1", name="camera", arguments="{}"
                            )
                        ]
                    )
                ]
            ),
            second_response,
        ]
    )
    agent = Agent(model="test-model", system_prompt="You are helpful.", tools=[tool])
    agent.client = make_fake_client(create)

    events = collect_chat(agent, [{"role": "user", "content": "look"}])

    assert (
        "".join(event.content for event in events if event.content)
        == "I can see the image."
    )


def test_agent_transcribes_voice_message_and_appends_user_text(monkeypatch):
    class FakeSegment:
        def __init__(self, text):
            self.text = text

    class FakeWhisperModel:
        def transcribe(self, _audio):
            return iter([FakeSegment("transcribed"), FakeSegment("text")]), {}

    faster_whisper_module = types.ModuleType("faster_whisper")
    faster_whisper_module.WhisperModel = lambda _name: FakeWhisperModel()
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper_module)
    audio_module = types.ModuleType("fury.utils.audio")
    audio_module.load_audio = lambda *args, **kwargs: (np.zeros(4), 16000)
    monkeypatch.setitem(sys.modules, "fury.utils.audio", audio_module)

    agent = Agent(model="test-model", system_prompt="You are helpful.")

    history = agent.add_voice_message_to_history([], "ZmFrZQ==")

    assert history == [{"role": "user", "content": "transcribed text"}]


def test_agent_speak_raises_when_reference_audio_or_text_missing():
    agent = Agent(model="test-model", system_prompt="You are helpful.")

    with pytest.raises(ValueError, match="ref_audio_path"):
        agent.speak(text="hello", ref_text="ref")

    with pytest.raises(ValueError, match="ref_text"):
        agent.speak(text="hello", ref_text="", ref_audio_path="ref.wav")


def test_agent_speak_initializes_tts_backend_once_and_reuses_it(monkeypatch):
    calls = []

    class FakeNeuTTSMinimal:
        def __init__(self, backbone_path, codec_path):
            calls.append(("init", backbone_path, codec_path))

        def infer_stream(self, text, ref_audio_path, ref_text):
            calls.append(("infer", text, ref_audio_path, ref_text))
            return iter([np.array([0.1, 0.2], dtype=np.float32)])

    module = types.ModuleType("fury.utils.neutts_minimal")
    module.NeuTTSMinimal = FakeNeuTTSMinimal
    monkeypatch.setitem(sys.modules, "fury.neutts_minimal", module)

    agent = Agent(model="test-model", system_prompt="You are helpful.")

    first = list(agent.speak(text="one", ref_text="ref", ref_audio_path="ref.wav"))
    second = list(agent.speak(text="two", ref_text="ref", ref_audio_path="ref.wav"))

    assert len([call for call in calls if call[0] == "init"]) == 1
    assert [call for call in calls if call[0] == "infer"] == [
        ("infer", "one", "ref.wav", "ref"),
        ("infer", "two", "ref.wav", "ref"),
    ]
    assert len(first) == 1
    assert len(second) == 1
