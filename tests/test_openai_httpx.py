import asyncio

from fury.openai_httpx import (
    AsyncStreamChatCompletions,
    ChatCompletionChunk,
    RetryConfig,
)


class StubResponse:
    def __init__(self, lines=None):
        self._lines = list(lines or [])
        self.closed = False

    def aiter_lines(self):
        async def iterator():
            for line in self._lines:
                yield line

        return iterator()

    async def aclose(self):
        self.closed = True


def make_stream() -> AsyncStreamChatCompletions:
    return AsyncStreamChatCompletions(
        client=None,
        url="https://example.invalid/v1/chat/completions",
        payload={},
        headers={},
        retry_config=RetryConfig(),
    )


def test_stream_retries_before_asserting_when_open_stream_defers_reconnect():
    stream = make_stream()
    open_calls = 0

    async def fake_open_stream():
        nonlocal open_calls
        open_calls += 1
        if open_calls == 1:
            return

        stream._response = StubResponse(
            ['data: {"choices":[{"delta":{"content":"hello"}}]}']
        )
        stream._lines = stream._response.aiter_lines()

    stream._open_stream = fake_open_stream

    chunk = asyncio.run(stream.__anext__())

    assert isinstance(chunk, ChatCompletionChunk)
    assert chunk.choices[0].delta.content == "hello"
    assert open_calls == 2


def test_stream_raises_clear_error_when_stream_opens_without_sse_iterator():
    stream = make_stream()

    async def fake_open_stream():
        stream._response = StubResponse()
        stream._lines = None

    stream._open_stream = fake_open_stream

    try:
        asyncio.run(stream.__anext__())
    except RuntimeError as exc:
        assert "SSE iterator" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")
