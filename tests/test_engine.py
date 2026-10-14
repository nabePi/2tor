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
    voices = []

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

        def generate(self, text, voice=None):
            voices.append(voice)
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
        engine.command("voice", "Luna")
        engine.command("start")
        wait_for(lambda: engine.active and not engine.busy and len(engine.history) == 1)
        assert voices == ["Luna", "Luna"]
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


def test_preview_is_keyless_local_and_reuses_tts(monkeypatch):
    import twotor.engine as module

    created = []
    voices = []
    playback = []

    class TTS:
        def __init__(self, *args):
            created.append(self)

        def generate(self, text, voice=None):
            voices.append(voice)
            return np.zeros(2400, dtype=np.float32)

    def unexpected(*args, **kwargs):
        raise AssertionError("Preview must never use STT, the microphone, or DeepSeek")

    monkeypatch.setattr(module, "SpeechToText", unexpected)
    monkeypatch.setattr(module, "DeepSeek", unexpected)
    monkeypatch.setattr(module, "TextToSpeech", TTS)
    monkeypatch.setattr(Engine, "_open_capture", unexpected)
    monkeypatch.setattr(module.sd, "stop", lambda: None)
    monkeypatch.setattr(module.sd, "play", lambda *a, **kw: playback.append(kw))
    monkeypatch.setattr(module.sd, "get_stream", lambda: type("Stream", (), {"active": False})())
    engine = Engine(Config(output_device="headphones"))
    try:
        engine.command("preview", "Luna")
        wait_for(lambda: len(playback) == 1 and not engine.busy)
        assert engine.voice == "Luna"
        assert not engine.active and not engine.listen.is_set()
        assert engine.stt is None and engine.capture is None
        assert engine.history == []
        engine.command("preview", "Leo")
        wait_for(lambda: len(playback) == 2 and not engine.busy)
        assert voices == ["Luna", "Leo"]
        assert len(created) == 1
        assert playback[0]["device"] == "headphones"
        assert engine.history == []
    finally:
        engine.command("quit")
        engine.thread.join(timeout=3)


def test_stop_during_preview_loading_prevents_late_playback(monkeypatch):
    import twotor.engine as module

    loading = threading.Event()
    release = threading.Event()
    playback = []

    class TTS:
        def __init__(self, *args):
            loading.set()
            release.wait(2)

        def generate(self, text, voice=None):
            return np.zeros(2400, dtype=np.float32)

    monkeypatch.setattr(module, "TextToSpeech", TTS)
    monkeypatch.setattr(module.sd, "stop", lambda: None)
    monkeypatch.setattr(module.sd, "play", lambda *a, **kw: playback.append(kw))
    engine = Engine(Config())
    try:
        engine.command("preview", "Bella")
        assert loading.wait(2)
        engine.command("pause")
        wait_for(lambda: engine.generation == 2)
        release.set()
        wait_for(lambda: not engine.busy)
        assert playback == []
        assert not engine.active and not engine.listen.is_set()
        assert engine.history == []
    finally:
        release.set()
        engine.command("quit")
        engine.thread.join(timeout=3)


def test_preview_during_session_restores_listening(monkeypatch):
    import twotor.engine as module

    played = threading.Event()

    class STT:
        def reset(self):
            pass

        def close(self):
            pass

    class TTS:
        def __init__(self, *args):
            pass

        def generate(self, text, voice=None):
            assert not engine.listen.is_set()
            return np.zeros(2400, dtype=np.float32)

    monkeypatch.setattr(module, "TextToSpeech", TTS)
    monkeypatch.setattr(module.sd, "play", lambda *a, **kw: played.set())
    monkeypatch.setattr(module.sd, "stop", lambda: None)
    monkeypatch.setattr(module.sd, "get_stream", lambda: type("Stream", (), {"active": False})())
    engine = Engine(Config())
    try:
        engine.stt = STT()
        engine.active = True
        engine.listen.set()
        engine.command("preview", "Hugo")
        assert played.wait(2)
        wait_for(lambda: not engine.busy)
        assert engine.listen.is_set() and engine.active
        assert engine.history == []
        engine.command("focus", True)
        wait_for(lambda: engine.focus)
        played.clear()
        engine.command("preview", "Hugo")
        wait_for(lambda: engine.commands.empty())
        assert not played.is_set()
    finally:
        engine.command("quit")
        engine.thread.join(timeout=3)
