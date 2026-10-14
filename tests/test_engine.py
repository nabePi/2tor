import threading
import time

import numpy as np

from twotor.config import Config
from twotor.engine import Engine


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Engine did not reach the expected state")


def test_pipeline_completes_without_ui_and_focus_cancels(monkeypatch):
    import twotor.engine as module

    cancelled = threading.Event()

    class STT:
        def __init__(self, *args):
            pass

        def reset(self):
            pass

        def close(self):
            pass

    class TTS:
        def __init__(self, *args):
            pass

        def generate(self, text):
            return np.zeros(2400, dtype=np.float32)

    class LLM:
        def __init__(self, *args):
            pass

        def stream(self, history, instruction, cancel):
            if not history:
                yield "Hello. How are you?"
            else:
                while not cancel.wait(0.01):
                    pass
                cancelled.set()

    monkeypatch.setattr(module, "SpeechToText", STT)
    monkeypatch.setattr(module, "TextToSpeech", TTS)
    monkeypatch.setattr(module, "DeepSeek", LLM)
    monkeypatch.setattr(module.sd, "play", lambda *a, **kw: None)
    monkeypatch.setattr(module.sd, "stop", lambda: None)
    monkeypatch.setattr(module.sd, "get_stream", lambda: type("Stream", (), {"active": False})())
    monkeypatch.setattr(Engine, "_open_capture", lambda self: None)
    engine = Engine(Config(api_key="test"))
    try:
        engine.command("start")
        wait_for(lambda: engine.active and not engine.busy and len(engine.history) == 1)
        assert engine.listen.is_set()
        engine.command("text", "I am working.")
        wait_for(lambda: engine.busy)
        assert not engine.listen.is_set()
        engine.command("focus", True)
        wait_for(lambda: engine.focus and not engine.busy)
        assert cancelled.is_set()
        assert not engine.listen.is_set()
        engine.command("focus", False)
        wait_for(lambda: engine.listen.is_set())
        engine.command("pause")
        wait_for(lambda: not engine.active)
        assert not engine.listen.is_set()
    finally:
        engine.command("quit")
        engine.thread.join(timeout=3)
        assert not engine.thread.is_alive()
