import queue
import threading

import pytest

from twotor.config import Config
from twotor.conversation import IdlePolicy, is_thinking_sound, system_prompt
from twotor.engine import Engine


@pytest.mark.parametrize("text", ["hmm", "hemm...", "enggg", "uh, um", "Hmm hemm enggg!", "eh"])
def test_standalone_thinking_sounds(text):
    assert is_thinking_sound(text)


@pytest.mark.parametrize(
    "text", ["egg", "enggak", "I am", "hmm I need help", "eh saya lupa", "123", ""]
)
def test_real_content_is_kept(text):
    assert not is_thinking_sound(text)


@pytest.fixture
def turn(monkeypatch):
    # Exercise the real owner-thread callbacks with a deterministic clock and no devices.
    engine = Engine.__new__(Engine)
    engine.listen = threading.Event()
    engine.listen.set()
    engine.active = True
    engine.testing = engine.busy = engine.focus = engine.speaking = False
    engine.pending = ""
    engine.pending_at = engine.speech_at = 0
    engine.end_pause = Config.stt_end_pause
    engine.policy = IdlePolicy()
    engine.audio = queue.Queue()
    engine.commands = queue.Queue()
    engine.emit = lambda *args: None
    accepted = []
    engine._accept_text = accepted.append
    clock = [0.0]
    monkeypatch.setattr("twotor.engine.time.monotonic", lambda: clock[0])
    return engine, clock, accepted


def test_resumed_speech_and_fillers_extend_five_second_wait(turn):
    engine, clock, accepted = turn
    engine._transcript("transcript", "I want")
    engine._commit_pending(4.9)
    assert not accepted
    clock[0] = 4.9
    engine._transcript("speech", "")
    engine._commit_pending(10)
    assert not accepted  # Active speech must never be committed.
    clock[0] = 6
    engine._transcript("transcript", "hemm enggg")
    engine._commit_pending(10.99)
    assert not accepted
    clock[0] = 10.99
    engine._transcript("transcript", "to learn. Saya lupa katanya.")
    engine._commit_pending(15.98)
    assert not accepted
    engine._commit_pending(16)
    assert accepted == ["I want to learn. Saya lupa katanya."]
    assert engine.commands.empty()  # No stale queued reply after speech resumes.


def test_empty_transcript_extends_pending_turn_and_fillers_alone_never_reply(turn):
    engine, clock, accepted = turn
    engine._transcript("transcript", "hmm")
    engine._commit_pending(6)
    assert not accepted
    engine._transcript("transcript", "I think")
    clock[0] = 4
    engine._transcript("transcript", "")
    engine._commit_pending(8.9)
    assert not accepted
    engine._commit_pending(9)
    assert accepted == ["I think"]
    assert engine.policy.last_activity == 4


def test_buffered_audio_and_stop_commands_prevent_commit(turn):
    engine, _, accepted = turn
    engine._transcript("transcript", "Hello")
    engine.audio.put("unprocessed audio")
    engine._commit_pending(6)
    assert not accepted
    engine.audio.get()
    engine.commands.put(("pause", None))
    engine._commit_pending(6)
    assert not accepted
    engine.commands.get()
    engine._commit_pending(6)
    assert accepted == ["Hello"]


def test_defaults_and_bilingual_tutoring_prompt():
    assert Config.stt_model == "whisper_small"
    assert Config.stt_end_pause == 5
    prompt = system_prompt("gentle")
    assert "Understand both English and Indonesian" in prompt
    assert "briefly help in Indonesian" in prompt
    assert "Never require them to repeat" in prompt
