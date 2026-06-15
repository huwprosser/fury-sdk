import asyncio

from conftest import FakeCompletion, FakeDelta, FakeToolCallChunk, SequencedCreate, make_fake_client

from fury import Agent, create_tool
from fury.utils.validation import validate_history


def _collect_chat(agent, history):
    async def run():
        events = []
        async for event in agent.runner().chat(history):
            events.append(event)
        return events

    return asyncio.run(run())


def _make_tool_agent():
    def add(a, b, emit=None):
        if emit:
            emit({"id": "ui-1", "title": "Adding", "type": "tool_call"})
        return a + b

    tool = create_tool(
        id="add",
        description="Add two numbers.",
        execute=add,
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        output_schema={"type": "number"},
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
                                arguments='{"a":2,"b":3}',
                            )
                        ]
                    )
                ]
            ),
            FakeCompletion([FakeDelta(content="Done")]),
        ]
    )
    agent = Agent(model="test-model", system_prompt="You are helpful.", tools=[tool])
    agent.client = make_fake_client(create)
    return agent


def test_tool_call_events_include_id_and_status_and_transcript_order():
    agent = _make_tool_agent()
    history = [{"role": "user", "content": "add"}]

    events = _collect_chat(agent, history)

    tool_events = [event.tool_call for event in events if event.tool_call]
    assert [(event.id, event.status) for event in tool_events] == [
        ("call_1", "started"),
        ("call_1", "completed"),
    ]

    deltas = [event.history_delta for event in events if event.history_delta]
    assert [delta.kind for delta in deltas] == [
        "assistant_tool_calls",
        "tool_result",
        "assistant_final",
    ]

    assert deltas[0].message == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a":2,"b":3}'},
            }
        ],
    }
    assert deltas[1].message == {
        "tool_call_id": "call_1",
        "role": "tool",
        "name": "add",
        "content": "5",
    }
    assert deltas[1].metadata == {"raw_result": 5}
    assert deltas[2].message == {"role": "assistant", "content": "Done"}

    validate_history([*history, *(delta.message for delta in deltas)])


def test_runner_complete_collects_content_transcript_tool_and_ui_events():
    async def run():
        agent = _make_tool_agent()
        return await agent.runner().complete([{"role": "user", "content": "add"}])

    result = asyncio.run(run())

    assert result.content == "Done"
    assert [message["role"] for message in result.transcript] == [
        "assistant",
        "tool",
        "assistant",
    ]
    assert [(event.id, event.status) for event in result.tool_events] == [
        ("call_1", "started"),
        ("call_1", "completed"),
    ]
    assert [(event.id, event.tool_call_id) for event in result.ui_events] == [
        ("ui-1", "call_1")
    ]
    validate_history([{"role": "user", "content": "add"}, *result.transcript])
