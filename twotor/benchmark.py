"""Repeatable local voice smoke/soak test; never opens the microphone or calls an API."""

import argparse
import json
import os
import threading
import time
from pathlib import Path

from .config import Config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--output", default="artifacts/benchmark.json")
    parser.add_argument("--env")
    args = parser.parse_args(argv)
    cfg = Config.load(args.env)
    os.environ.setdefault("OMP_NUM_THREADS", str(cfg.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import platform

    import numpy as np
    import psutil

    from .engine import memory_mb
    from .voice import SpeechToText, TextToSpeech

    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "stt_model": cfg.stt_model,
        "tts_model": cfg.tts_model,
        "tts_threads": cfg.threads,
        "turns": [],
        "note": "Synthetic English speech, no microphone, UI, network LLM or speaker playback. "
        "Measure again on Ubuntu and with your own accent.",
    }
    peak = [memory_mb()]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.05):
            peak[0] = max(peak[0], memory_mb())

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    stt = None
    try:
        started = time.monotonic()
        tts = TextToSpeech(cfg)
        report["tts_load_seconds"] = time.monotonic() - started
        transcript = []

        def event(name, text):
            if name == "transcript":
                transcript.append(text)

        started = time.monotonic()
        stt = SpeechToText(cfg, event)
        report["stt_load_seconds"] = time.monotonic() - started
        report["loaded_rss_mb"] = memory_mb()
        phrases = [
            "How is your work going today?",
            "I am working on a small project.",
            "Could you help me practice my English?",
        ]
        cpu_start = psutil.Process().cpu_times()
        started = time.monotonic()
        for turn in range(args.turns):
            phrase = phrases[turn % len(phrases)]
            transcript.clear()
            t = time.monotonic()
            audio = tts.generate(phrase)
            tts_seconds = time.monotonic() - t
            duration = len(audio) / 24000
            # Feed a synthetic recording in 100ms chunks, with VAD trailing silence.
            audio = np.concatenate(
                [np.zeros(4800, dtype=np.float32), audio, np.zeros(36000, dtype=np.float32)]
            )
            t = time.monotonic()
            for offset in range(0, len(audio), 2400):
                stt.feed(audio[offset : offset + 2400], 24000)
            if not transcript:
                raise RuntimeError("STT returned no completed transcript for synthesized speech")
            report["turns"].append(
                {
                    "input": phrase,
                    "transcript": list(transcript),
                    "audio_seconds": duration,
                    "tts_seconds": tts_seconds,
                    "stt_compute_seconds": time.monotonic() - t,
                    "tts_realtime_factor": tts_seconds / duration,
                    "rss_mb": memory_mb(),
                }
            )
            stt.reset()
        cpu_end = psutil.Process().cpu_times()
        report["wall_seconds"] = time.monotonic() - started
        report["cpu_seconds"] = cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system
        report["end_rss_mb"] = memory_mb()
        report["peak_rss_mb"] = max(peak[0], memory_mb())
        report["within_memory_target"] = report["peak_rss_mb"] < cfg.memory_limit_mb
        report["tts_faster_than_realtime"] = all(
            turn["tts_realtime_factor"] < 1 for turn in report["turns"]
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "turns"}, indent=2))
        print(f"Full results: {output.resolve()}")
        if not report["within_memory_target"] or not report["tts_faster_than_realtime"]:
            raise SystemExit(1)
    finally:
        stop.set()
        monitor.join()
        if stt:
            stt.close()


if __name__ == "__main__":
    main()
