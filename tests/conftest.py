import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from fury.agent import Agent


class FakeDelta:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class FakeChoice:
    def __init__(self, delta):
        self.delta = delta


class FakeChunk:
    def __init__(self, delta):
        self.choices = [FakeChoice(delta)]


class FakeCompletion:
    def __init__(self, deltas):
        self._deltas = list(deltas)
        self._index = 0

    def __aiter__(self):
        self._index = 0
        return self

    async def __anext__(self):
        if self._index >= len(self._deltas):
            raise StopAsyncIteration
        delta = self._deltas[self._index]
        self._index += 1
        return FakeChunk(delta)


class FakeToolFunction:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class FakeToolCallChunk:
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = FakeToolFunction(name=name, arguments=arguments)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeResponseChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeResponseChoice(content)]


class SequencedCreate:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("Unexpected completion request")
        response = self._responses.pop(0)
        return response(kwargs) if callable(response) else response


@pytest.fixture(autouse=True)
def suppress_agent_banner(monkeypatch):
    monkeypatch.setattr(Agent, "show_yourself", lambda self: None)


@pytest.fixture
def coding_assistant_module(monkeypatch):
    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda: None
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_module)

    pydantic_module = types.ModuleType("pydantic")

    class BaseModel:
        @classmethod
        def model_json_schema(cls):
            return {"type": "object", "properties": {}, "required": []}

        @classmethod
        def schema(cls):
            return cls.model_json_schema()

    pydantic_module.BaseModel = BaseModel
    monkeypatch.setitem(sys.modules, "pydantic", pydantic_module)

    module_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "coding-assistant"
        / "coding_assistant.py"
    )
    spec = importlib.util.spec_from_file_location("test_coding_assistant_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_fake_client(create):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
