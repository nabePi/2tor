import asyncio
import json
import threading
import time

import httpx
import pytest

from twotor.config import Config
from twotor.llm import DeepSeek


def test_stream_uses_key_non_thinking_and_bounded_history():
    def handler(request):
        assert request.headers["authorization"] == "Bearer test"
        payload = json.loads(request.content)
        assert payload["thinking"] == {"type": "disabled"}
        assert len(payload["messages"]) == 22
        return httpx.Response(
            200, text='data: {"choices":[{"delta":{"content":"Hello."}}]}\n\ndata: [DONE]\n\n'
        )

    llm = DeepSeek(Config(api_key="test"), httpx.MockTransport(handler))
    history = [{"role": "user", "content": "Hello"}] * 50
    assert list(llm.stream(history, "Open a topic", threading.Event())) == ["Hello."]


def test_cancelled_stream_emits_nothing():
    cancel = threading.Event()
    cancel.set()
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, text='data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n')
    )
    assert list(DeepSeek(Config(api_key="test"), transport).stream([], None, cancel)) == []


def test_http_error_never_exposes_provider_response_or_key():
    transport = httpx.MockTransport(lambda _: httpx.Response(401, text="secret provider data"))
    with pytest.raises(RuntimeError, match="API key invalid"):
        list(DeepSeek(Config(api_key="secret"), transport).stream([], None, threading.Event()))


def test_incomplete_stream_is_reported():
    transport = httpx.MockTransport(lambda _: httpx.Response(200, text="data: {}\n\n"))
    with pytest.raises(RuntimeError, match="before the reply was complete"):
        list(DeepSeek(Config(api_key="test"), transport).stream([], None, threading.Event()))


def test_cancel_interrupts_a_slow_network_read():
    closed = threading.Event()

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(10)
            yield b"data: [DONE]\n\n"

        async def aclose(self):
            closed.set()

    transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=SlowStream()))
    cancel = threading.Event()
    timer = threading.Timer(0.1, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        assert list(DeepSeek(Config(api_key="test"), transport).stream([], None, cancel)) == []
        assert time.monotonic() - started < 1
        assert closed.is_set()
    finally:
        timer.cancel()
