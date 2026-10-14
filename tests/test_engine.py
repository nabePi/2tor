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
    models = []

    class STT:
        def __init__(self, *args):
            models.append(args[0].stt_model)

        def reset(self):
            pass

        def close(self):
            pass

    class TTS:
        def __init__(self, *args):
            pass

        def generate(self, text, voice=None, speed=None):
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
    engine = Engine(
        Config(
            api_key="test",
            stt_model="small_streaming",
            tts_model="KittenML/kitten-tts-nano-0.8-int8",
            voice="Jasper",
        )
    )
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
        engine.command("stt_model", "tiny_streaming")
        wait_for(lambda: engine.stt_model == "tiny_streaming")
        assert engine.stt is None
        engine.command("start")
        wait_for(lambda: engine.active)
        assert models == ["small_streaming", "tiny_streaming"]
        engine.command("stt_model", "small_streaming")
        wait_for(lambda: engine.commands.empty())
        assert engine.stt_model == "tiny_streaming"  # Reject switching a live microphone.
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

        def generate(self, text, voice=None, speed=None):
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
    engine = Engine(
        Config(
            output_device="headphones",
            tts_model="KittenML/kitten-tts-nano-0.8-int8",
            voice="Jasper",
        )
    )
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

        def generate(self, text, voice=None, speed=None):
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

        def generate(self, text, voice=None, speed=None):
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
        engine.command("preview", "Heart")
        assert played.wait(2)
        wait_for(lambda: not engine.busy)
        assert engine.listen.is_set() and engine.active
        assert engine.history == []
        engine.command("focus", True)
        wait_for(lambda: engine.focus)
        played.clear()
        engine.command("preview", "Heart")
        wait_for(lambda: engine.commands.empty())
        assert not played.is_set()
    finally:
        engine.command("quit")
        engine.thread.join(timeout=3)


def test_mic_test_needs_no_key_and_preserves_actual_recognized_words(monkeypatch):
    import twotor.engine as module

    class STT:
        def __init__(self, *args):
            pass

        def reset(self):
            pass

        def close(self):
            pass

    def unexpected(*args, **kwargs):
        raise AssertionError("Mic test must not load TTS or call DeepSeek")

    monkeypatch.setattr(module, "SpeechToText", STT)
    monkeypatch.setattr(module, "TextToSpeech", unexpected)
    monkeypatch.setattr(module, "DeepSeek", unexpected)
    monkeypatch.setattr(module.sd, "stop", lambda: None)
    monkeypatch.setattr(Engine, "_open_capture", lambda self: None)
    engine = Engine(Config())
    try:
        engine.command("mic_test")
        wait_for(lambda: engine.testing and engine.listen.is_set())
        engine._transcript("transcript", "Best, best, best.")
        assert engine.history == [] and engine.pending == ""
        events = []
        while not engine.events.empty():
            events.append(engine.events.get_nowait())
        assert ("user", "Best, best, best.", None) in events
        engine.command("pause")
        wait_for(lambda: not engine.testing and not engine.active)
    finally:
        engine.command("quit")
        engine.thread.join(timeout=3)


def test_next_sentence_is_synthesized_while_previous_audio_is_playing(monkeypatch):
    import twotor.engine as module

    first_playing = threading.Event()
    second_ready = threading.Event()
    playback = []

    class STT:
        def __init__(self, *args):
            pass

        def reset(self):
            pass

        def close(self):
            pass

    class TTS:
        def __init__(self, *args):
            self.calls = 0

        def generate(self, text, voice=None, speed=None):
            self.calls += 1
            if self.calls == 2:
                assert first_playing.wait(2)
                second_ready.set()
            return np.full(24, self.calls, dtype=np.float32)

    class LLM:
        def __init__(self, *args):
            pass

        def stream(self, *args):
            yield "Hello. How are you?"

    class Stream:
        @property
        def active(self):
            return len(playback) == 1 and not second_ready.is_set()

    def play(audio, *args, **kwargs):
        playback.append(int(audio[0]))
        first_playing.set()

    monkeypatch.setattr(module, "SpeechToText", STT)
    monkeypatch.setattr(module, "TextToSpeech", TTS)
    monkeypatch.setattr(module, "DeepSeek", LLM)
    monkeypatch.setattr(module.sd, "stop", lambda: None)
    monkeypatch.setattr(module.sd, "play", play)
    monkeypatch.setattr(module.sd, "get_stream", Stream)
    monkeypatch.setattr(Engine, "_open_capture", lambda self: None)
    engine = Engine(Config(api_key="test"))
    try:
        engine.command("start")
        wait_for(lambda: len(engine.history) == 1 and not engine.busy)
        assert second_ready.is_set() and playback == [1, 2]
        assert engine.listen.is_set()
    finally:
        engine.command("quit")
        engine.thread.join(timeout=3)


def test_tts_switch_reloads_only_selected_engine_and_updates_voices(monkeypatch):
    import twotor.engine as module

    loaded = []
    played = []

    class TTS:
        def __init__(self, config):
            loaded.append((config.tts_model, config.voice))

        def generate(self, text, voice=None, speed=None):
            played.append((voice, speed))
            return np.zeros(24, dtype=np.float32)

    monkeypatch.setattr(module, "TextToSpeech", TTS)
    monkeypatch.setattr(module.sd, "stop", lambda: None)
    monkeypatch.setattr(module.sd, "play", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.sd, "get_stream", lambda: type("Stream", (), {"active": False})())
    engine = Engine(Config(tts_model="KittenML/kitten-tts-nano-0.8-int8", voice="Jasper"))
    try:
        engine.command("tts_model", "kokoro_int8")
        wait_for(lambda: engine.tts_model == "kokoro_int8")
        assert engine.tts is None and engine.voice == "Heart"
        assert "Fenrir" in engine.voices and "Jasper" not in engine.voices
        engine.command("preview", "Heart")
        wait_for(lambda: played and not engine.busy)
        assert loaded == [("kokoro_int8", "Heart")]
        assert played == [("Heart", 1.05)]
    finally:
        engine.command("quit")
        engine.thread.join(timeout=3)
