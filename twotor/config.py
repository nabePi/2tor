import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

VOICES = ("Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo")
KOKORO_VOICES = {
    "Heart": "af_heart",
    "Bella": "af_bella",
    "Nicole": "af_nicole",
    "Sarah": "af_sarah",
    "Adam": "am_adam",
    "Michael": "am_michael",
    "Fenrir": "am_fenrir",
    "Emma": "bf_emma",
}
STT_MODELS = (
    "whisper_small",
    "whisper_base",
    "whisper_small_en",
    "whisper_base_en",
    "tiny_streaming",
    "small_streaming",
)
EDGE_VOICES = {
    "Jenny": "en-US-JennyNeural",
    "Emma": "en-US-EmmaMultilingualNeural",
    "Aria": "en-US-AriaNeural",
    "Guy": "en-US-GuyNeural",
    "Andrew": "en-US-AndrewMultilingualNeural",
    "Brian": "en-US-BrianMultilingualNeural",
    "Ava": "en-US-AvaMultilingualNeural",
    "Christopher": "en-US-ChristopherNeural",
    "Sonia": "en-GB-SoniaNeural",
    "Ryan": "en-GB-RyanNeural",
    "Natasha": "en-AU-NatashaNeural",
    "William": "en-AU-WilliamMultilingualNeural",
}


def voices_for_model(model):
    if model == "edge_tts":
        return tuple(EDGE_VOICES)
    return tuple(KOKORO_VOICES) if model == "kokoro_int8" else VOICES


def device(value: str):
    value = value.strip()
    return int(value) if value.isdigit() else value or None


@dataclass(frozen=True)
class Config:
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    stt_model: str = "whisper_small"
    stt_end_pause: float = 5.0
    stt_keyterms: tuple[str, ...] = ()
    stt_threads: int = 2
    tts_model: str = "kokoro_int8"
    voice: str = "Heart"
    speed: float = 1.05
    threads: int = 2
    input_device: object = None
    output_device: object = None
    proactive_seconds: int = 90
    save_history: bool = False
    memory_limit_mb: int = 2800
    correction: str = "gentle"

    @classmethod
    def load(cls, env_path=None):
        candidate = Path.cwd() / ".env"
        if not candidate.is_file():
            candidate = Path.home() / ".config/2tor/.env"
        load_dotenv(env_path or candidate, override=False)
        tts_model = os.getenv("TTS_MODEL", cls.tts_model)
        cfg = cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
            base_url=os.getenv("DEEPSEEK_BASE_URL", cls.base_url).rstrip("/"),
            model=os.getenv("DEEPSEEK_MODEL", cls.model),
            stt_model=os.getenv("STT_MODEL", cls.stt_model),
            stt_end_pause=float(os.getenv("STT_END_PAUSE_SECONDS", str(cls.stt_end_pause))),
            stt_threads=int(os.getenv("STT_THREADS", "2")),
            stt_keyterms=tuple(
                term.strip() for term in os.getenv("STT_KEYTERMS", "").split(",") if term.strip()
            ),
            tts_model=tts_model,
            voice=os.getenv(
                "TTS_VOICE",
                "Jenny"
                if tts_model == "edge_tts"
                else "Heart"
                if tts_model == "kokoro_int8"
                else "Jasper",
            ),
            speed=float(os.getenv("TTS_SPEED", str(cls.speed))),
            threads=int(os.getenv("TTS_THREADS", str(cls.threads))),
            input_device=device(os.getenv("INPUT_DEVICE", "")),
            output_device=device(os.getenv("OUTPUT_DEVICE", "")),
            proactive_seconds=int(os.getenv("PROACTIVE_SECONDS", "90")),
            save_history=os.getenv("SAVE_HISTORY", "0") == "1",
            memory_limit_mb=int(os.getenv("MEMORY_LIMIT_MB", "2800")),
            correction=os.getenv("CORRECTION_MODE", "gentle"),
        )
        if cfg.stt_model not in STT_MODELS:
            raise ValueError("Unknown STT_MODEL")
        if not 1 <= cfg.stt_threads <= 4:
            raise ValueError("STT_THREADS must be 1–4")
        if not 0.5 <= cfg.stt_end_pause <= 15:
            raise ValueError("STT_END_PAUSE_SECONDS must be 0.5–15")
        if len(cfg.stt_keyterms) > 50 or any(len(term) > 80 for term in cfg.stt_keyterms):
            raise ValueError("STT_KEYTERMS supports at most 50 terms, each up to 80 characters")
        if cfg.tts_model not in {
            "edge_tts",
            "kokoro_int8",
            "KittenML/kitten-tts-nano-0.8-int8",
            "KittenML/kitten-tts-micro-0.8",
            "KittenML/kitten-tts-mini-0.8",
        }:
            raise ValueError("TTS_MODEL must be Edge TTS or a supported lightweight ONNX model")
        if cfg.voice not in voices_for_model(cfg.tts_model):
            raise ValueError("Unknown TTS_VOICE")
        if not 0.7 <= cfg.speed <= 1.3 or not 1 <= cfg.threads <= 4:
            raise ValueError("TTS_SPEED must be 0.7–1.3; TTS_THREADS must be 1–4")
        if not 30 <= cfg.proactive_seconds <= 3600:
            raise ValueError("PROACTIVE_SECONDS must be 30–3600")
        if not 256 <= cfg.memory_limit_mb <= 3000:
            raise ValueError("MEMORY_LIMIT_MB must be 256–3000")
        if cfg.correction not in {"gentle", "off"}:
            raise ValueError("CORRECTION_MODE must be gentle or off")
        if not cfg.base_url.startswith("https://"):
            raise ValueError("DEEPSEEK_BASE_URL must use HTTPS")
        return cfg
