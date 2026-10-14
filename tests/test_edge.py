import asyncio
import io
import threading
import time

import av
import edge_tts
import numpy as np
import pytest

from twotor.config import Config, voices_for_model
from twotor.voice import EdgeSpeech, TextToSpeech, decode_edge_audio


def mp3_audio():
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp3") as container:
        stream = container.add_stream("libmp3lame", rate=24000)
        wave = (0.1 * np.sin(np.arange(4800) * 2 * np.pi * 440 / 24000)).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(wave[None, :], format="fltp", layout="mono")
        frame.sample_rate = 24000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


def test_mp3_decodes_to_mono_float32_24khz():
    audio = decode_edge_audio(mp3_audio())
    assert audio.ndim == 1 and audio.dtype == np.float32
    assert 0.18 <= len(audio) / 24000 <= 0.3
    assert np.isfinite(audio).all() and np.max(np.abs(audio)) > 0.01


def test_edge_voice_rate_and_no_constructor_network(monkeypatch):
    requests = []
    data = mp3_audio()

    class Communication:
        def __init__(self, text, voice, **options):
            requests.append((text, voice, options))

        async def stream(self):
            yield {"type": "SentenceBoundary"}
            yield {"type": "audio", "data": data}

    monkeypatch.setattr(edge_tts, "Communicate", Communication)
    tts = TextToSpeech(Config(tts_model="edge_tts", voice="Jenny"))
    assert isinstance(tts, EdgeSpeech) and requests == []
    audio = tts.generate("Hello there.", voice="Sonia", speed=1.1)
    assert requests[0][0:2] == ("Hello there.", "en-GB-SoniaNeural")
    assert requests[0][2]["rate"] == "+10%"
    assert len(audio) > 0


def test_edge_cancel_closes_slow_stream(monkeypatch):
    closed = threading.Event()

    class Communication:
        def __init__(self, *args, **options):
            pass

        async def stream(self):
            try:
                await asyncio.sleep(10)
                yield {"type": "audio", "data": b""}
            finally:
                closed.set()

    monkeypatch.setattr(edge_tts, "Communicate", Communication)
    cancel = threading.Event()
    timer = threading.Timer(0.1, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        audio = EdgeSpeech(Config(tts_model="edge_tts", voice="Jenny")).generate(
            "Hello.", cancel=cancel
        )
        assert len(audio) == 0 and closed.is_set()
        assert time.monotonic() - started < 1
    finally:
        timer.cancel()


def test_edge_network_failure_has_actionable_hint(monkeypatch):
    class Communication:
        def __init__(self, *args, **options):
            raise OSError("service error")

    monkeypatch.setattr(edge_tts, "Communicate", Communication)
    with pytest.raises(RuntimeError, match="Check internet or select Kokoro/Kitten"):
        EdgeSpeech(Config(tts_model="edge_tts", voice="Jenny")).generate("Hello.")


def test_edge_env_defaults_and_voice_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("TTS_MODEL", "edge_tts")
    monkeypatch.delenv("TTS_VOICE", raising=False)
    cfg = Config.load(tmp_path / "absent.env")
    assert cfg.voice == "Jenny" and "William" in voices_for_model("edge_tts")
    monkeypatch.setenv("TTS_VOICE", "Heart")
    with pytest.raises(ValueError, match="Unknown TTS_VOICE"):
        Config.load(tmp_path / "absent.env")
