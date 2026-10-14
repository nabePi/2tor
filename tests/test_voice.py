from types import SimpleNamespace

import numpy as np
from moonshine_voice import LineCompleted, LineStarted

from twotor.voice import SpeechToText


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
