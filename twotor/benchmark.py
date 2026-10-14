"""Voice smoke/soak test without microphone or DeepSeek. Edge TTS uses the internet."""

import argparse
import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path

from .config import STT_MODELS, Config, voices_for_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--output", default="artifacts/benchmark.json")
    parser.add_argument("--env")
    parser.add_argument("--stt-model", choices=STT_MODELS)
    parser.add_argument(
        "--tts-model", choices=["edge_tts", "kokoro_int8", "KittenML/kitten-tts-nano-0.8-int8"]
    )
    args = parser.parse_args(argv)
    cfg = Config.load(args.env)
    if args.stt_model:
        cfg = replace(cfg, stt_model=args.stt_model)
    if args.tts_model:
        cfg = replace(cfg, tts_model=args.tts_model, voice=voices_for_model(args.tts_model)[0])
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
        "tts_online": cfg.tts_model == "edge_tts",
        "note": "Synthetic English speech, no microphone, UI, network LLM or speaker playback. "
        "Edge TTS sends synthetic phrases to Microsoft if selected. "
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
            "Test, test, test.",
            "Can you hear me clearly?",
            "I think this task takes time.",
            "Best, best, best.",
            "The best way is to test it.",
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
