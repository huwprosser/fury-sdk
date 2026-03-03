import asyncio
from types import SimpleNamespace

from fury import Agent


def test_basic_chat_loop():
    agent = Agent(
        model="unsloth/GLM-4.6V-Flash-GGUF:Q8_0",
        system_prompt="You are a helpful assistant.",
    )

    async def run_chat():
        history = [
            {"role": "user", "content": "Hello! Please reply with a short greeting."}
        ]
        buffer = []

        async for event in agent.chat(history, reasoning=False):
            if event.content:
                buffer.append(event.content)

        return "".join(buffer).strip()

    response = asyncio.run(run_chat())
    assert response


def test_chat_prunes_incomplete_sentences():
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

    async def fake_create(**_kwargs):
        deltas = [
            FakeDelta(content="Hello there. This is"),
            FakeDelta(content=" a test!"),
            FakeDelta(content=" And another sentence."),
            FakeDelta(content=" Trailing fragment with no ending"),
        ]
        return FakeCompletion(deltas)

    agent = Agent(model="test-model", system_prompt="You are a helpful assistant.")
    agent.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )

    async def run_chat(prune_unfinished_sentences):
        history = [{"role": "user", "content": "Hi"}]
        chunks = []

        async for event in agent.chat(
            history,
            reasoning=False,
            prune_unfinished_sentences=prune_unfinished_sentences,
        ):
            if event.content:
                chunks.append(event.content)

        return chunks, "".join(chunks)

    pruned_chunks, pruned_response = asyncio.run(run_chat(True))
    unpruned_chunks, unpruned_response = asyncio.run(run_chat(False))

    # With pruning on, we do not emit anything until a full sentence boundary exists.
    assert pruned_chunks[0] == "Hello there. This is a test!"
    assert pruned_response == "Hello there. This is a test! And another sentence."

    # With pruning off, incomplete trailing text is streamed through.
    assert unpruned_response.endswith("Trailing fragment with no ending")
