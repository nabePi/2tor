from types import SimpleNamespace

import numpy as np
from moonshine_voice import LineCompleted, LineStarted

from twotor.voice import SpeechToText, WhisperSpeechToText


def test_completed_segment_does_not_discard_next_onset_in_same_audio_block():
    stt = SpeechToText.__new__(SpeechToText)
    events = []
    stt.callback = lambda name, text: events.append((name, text))

    class Stream:
        def __init__(self):
            self.calls = 0

        def add_audio(self, audio, rate):
            self.calls += 1
            if self.calls == 1:
                stt._event(LineCompleted(SimpleNamespace(text="I want"), 1))
                stt._event(LineStarted(SimpleNamespace(text="to test"), 1))
            else:
                stt._event(LineCompleted(SimpleNamespace(text="to test this"), 1))

    stream = Stream()
    stt.stream = stream
    stt.feed(np.zeros(1600), 16000)
    assert stt.stream is stream
    stt.feed(np.zeros(1600), 16000)
    assert events == [("transcript", "I want"), ("speech", ""), ("transcript", "to test this")]


def test_whisper_preserves_best_and_uses_english_beam_search():
    stt = WhisperSpeechToText.__new__(WhisperSpeechToText)
    output = []
    stt.callback = lambda name, text: output.append((name, text))
    stt.keyterms = None

    class Model:
        def transcribe(self, audio, **options):
            assert options["language"] == "en"
            assert options["beam_size"] == 5
            assert options["condition_on_previous_text"] is False
            assert options["hotwords"] is None
            return iter([SimpleNamespace(text="Best, best, best.", no_speech_prob=0.1)]), None

    stt.model = Model()
    stt._event(LineCompleted(SimpleNamespace(audio_data=[0.01] * 32000), 1))
    assert output[-1] == ("transcript", "Best, best, best.")


def test_whisper_silence_never_calls_decoder():
    stt = WhisperSpeechToText.__new__(WhisperSpeechToText)
    output = []
    stt.callback = lambda name, text: output.append((name, text))
    stt.model = None
    stt._event(LineCompleted(SimpleNamespace(audio_data=[0.0] * 32000), 1))
    assert output == [("transcript", "")]
