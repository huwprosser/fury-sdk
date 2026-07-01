import asyncio

import pytest
from conftest import FakeDelta, make_fake_client

from fury import Agent, StreamError


class Choice:
    def __init__(self, delta=None, finish_reason=None, error=None):
        self.delta = delta
        self.finish_reason = finish_reason
        if error is not None:
            self.error = error


class ChunkWithChoices:
    def __init__(self, choices):
        self.choices = choices


class ChunkWithError:
    def __init__(self, error):
        self.choices = None
        self.error = error


class CustomCompletion:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self):
        self._index = 0
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk

    async def aclose(self):
        self.closed = True


def _agent_for(chunks):
    async def create(**kwargs):
        return CustomCompletion(chunks)

    agent = Agent(model="test-model", system_prompt="You are helpful.", tools=[])
    agent.client = make_fake_client(create)
    return agent


def _collect(agent):
    async def run():
        events = []
        async for event in agent.runner().chat([{"role": "user", "content": "hi"}]):
            events.append(event)
        return events

    return asyncio.run(run())


def test_finish_reason_error_raises_stream_error():
    agent = _agent_for(
        [
            ChunkWithChoices([Choice(delta=FakeDelta(content="partial"))]),
            ChunkWithChoices([Choice(delta=FakeDelta(), finish_reason="error")]),
        ]
    )
    with pytest.raises(StreamError):
        _collect(agent)


def test_error_payload_chunk_raises_stream_error():
    agent = _agent_for([ChunkWithError({"message": "rate limited", "code": 429})])
    with pytest.raises(StreamError) as excinfo:
        _collect(agent)
    assert "rate limited" in str(excinfo.value)
    assert excinfo.value.code == 429


def test_normal_finish_reason_stop_does_not_raise():
    agent = _agent_for(
        [
            ChunkWithChoices([Choice(delta=FakeDelta(content="all good"))]),
            ChunkWithChoices([Choice(delta=FakeDelta(), finish_reason="stop")]),
        ]
    )
    events = _collect(agent)
    content = "".join(event.content for event in events if event.content)
    assert content == "all good"
