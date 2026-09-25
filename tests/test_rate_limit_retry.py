import asyncio
from unittest.mock import patch

import httpx
import pytest
from conftest import FakeCompletion, FakeDelta, make_fake_client
from openai import RateLimitError

from fury import Agent, runtime as rt


def _rate_limit_error():
    request = httpx.Request("POST", "http://test/v1/chat/completions")
    response = httpx.Response(429, request=request, text="rate limited")
    return RateLimitError("rate limited", response=response, body=None)


def _collect(agent):
    async def run():
        events = []
        async for event in agent.runner().chat([{"role": "user", "content": "hi"}]):
            events.append(event)
        return events

    return asyncio.run(run())


def _agent_for(create):
    agent = Agent(model="test-model", system_prompt="You are helpful.", tools=[])
    agent.client = make_fake_client(create)
    return agent


def test_transient_429_retried_then_succeeds():
    calls = {"n": 0}

    async def create(**kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _rate_limit_error()
        return FakeCompletion([FakeDelta(content="hello")])

    agent = _agent_for(create)
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    with patch.object(rt.asyncio, "sleep", fake_sleep):
        events = _collect(agent)

    assert "".join(e.content for e in events if e.content) == "hello"
    assert calls["n"] == 3
    assert sleeps == [2.0, 2.0]


def test_persistent_429_fails_after_five_retries():
    calls = {"n": 0}

    async def create(**kwargs):
        calls["n"] += 1
        raise _rate_limit_error()

    agent = _agent_for(create)
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    with patch.object(rt.asyncio, "sleep", fake_sleep):
        with pytest.raises(RateLimitError):
            _collect(agent)

    # 1 initial attempt + 5 retries, 2 seconds apart.
    assert calls["n"] == 6
    assert sleeps == [2.0] * 5


def test_non_429_errors_are_not_retried():
    calls = {"n": 0}

    async def create(**kwargs):
        calls["n"] += 1
        raise ValueError("boom")

    agent = _agent_for(create)
    with patch.object(rt.asyncio, "sleep") as sleep_mock:
        _collect(agent)

    assert calls["n"] == 1
    sleep_mock.assert_not_called()
