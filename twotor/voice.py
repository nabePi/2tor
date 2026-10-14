"""Local STT, local ONNX TTS, and opt-in online Edge TTS. No saved audio."""

import json
import os
from pathlib import Path

import numpy as np

from .config import EDGE_VOICES, KOKORO_VOICES, voices_for_model


class SpeechToText:
    def __new__(cls, config=None, callback=None, progress=lambda *args: None):
        if config and config.stt_model.startswith("whisper_"):
            return WhisperSpeechToText(config, callback, progress)
        return super().__new__(cls)

    def __init__(self, config, callback, progress=lambda *args: None):
        os.environ.setdefault("MOONSHINE_ORT_SINGLE_THREAD", "1")
        from moonshine_voice import ModelArch, Transcriber
        from moonshine_voice.download import get_model_for_language

        arch = {
            "tiny_streaming": ModelArch.TINY_STREAMING,
            "small_streaming": ModelArch.SMALL_STREAMING,
        }[config.stt_model]
        path, arch = get_model_for_language("en", arch, on_progress=progress)
        self.transcriber = Transcriber(
            path,
            arch,
            options={
                "return_audio_data": "false",
                "word_timestamps": "false",
                "vad_max_segment_duration": 25.0,
                "ort_providers": "cpu",
            },
        )
        self.callback = callback
        if config.stt_keyterms:
            self.transcriber.set_keyterms(config.stt_keyterms)
        self.stream = None
        self.reset()

    def reset(self):
        if self.stream is not None:
            self.stream.remove_all_listeners()
            self.stream.close()
        self.stream = self.transcriber.create_stream(update_interval=0.5)
        self.stream.add_listener(self._event)
        self.stream.start()

    def _event(self, event):
        from moonshine_voice import LineCompleted, LineStarted, LineTextChanged

        if isinstance(event, LineStarted):
            self.callback("speech", "")
        elif isinstance(event, LineTextChanged):
            self.callback("partial", event.line.text)
        elif isinstance(event, LineCompleted):
            self.callback("transcript", event.line.text)

    def feed(self, audio, rate):
        self.stream.add_audio(audio, rate)
        # A completed VAD segment may share a block with the next speech onset.
        # Keep the stream until the engine ends the user's turn, preserving
        # look-behind audio and context across short pauses.

    def close(self):
        if self.stream:
            self.stream.remove_all_listeners()
            self.stream.close()
        self.transcriber.close()


class TextToSpeech:
    def __new__(cls, config):
        if config.tts_model == "edge_tts":
            return EdgeSpeech(config)
        if config.tts_model == "kokoro_int8":
            return KokoroSpeech(config)
        return super().__new__(cls)

    def __init__(self, config):
        import onnxruntime as ort
        import phonemizer
        from huggingface_hub import hf_hub_download
        from kittenml.kittentts_legacy.onnx_model import KittenTTS_1_Onnx, TextCleaner
        from kittenml.preprocess import TextPreprocessor

        # Upstream's pinned constructor lacks SessionOptions. Initialize its
        # inference helper explicitly so CPU threading is bounded from the start.
        def fetch(name):
            return hf_hub_download(repo_id=config.tts_model, filename=name)

        info = json.loads(Path(fetch("config.json")).read_text())
        if info.get("type") not in {"ONNX1", "ONNX2"}:
            raise ValueError("Only lightweight ONNX KittenTTS models are supported")
        options = ort.SessionOptions()
        options.intra_op_num_threads = config.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.enable_cpu_mem_arena = False
        self.model = KittenTTS_1_Onnx.__new__(KittenTTS_1_Onnx)
        self.model.model_path = fetch(info["model_file"])
        self.model.session = ort.InferenceSession(
            self.model.model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        with np.load(fetch(info["voices"])) as archive:
            self.model.voices = {key: archive[key] for key in archive.files}
        self.model.phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )
        self.model.text_cleaner = TextCleaner()
        self.model.speed_priors = info.get("speed_priors", {})
        self.model.voice_aliases = info.get("voice_aliases", {})
        self.model.available_voices = list(self.model.voices)
        self.model.preprocessor = TextPreprocessor(remove_punctuation=False)
        self.config = config
        # Warm up without playing anything.
        self.generate("Hello.")

    def generate(self, text, voice=None, speed=None, cancel=None):
        voice = voice or self.config.voice
        if voice not in voices_for_model(self.config.tts_model):
            raise ValueError("Unknown TTS voice")
        return np.asarray(
            self.model.generate(
                text, voice=voice, speed=speed or self.config.speed, clean_text=True
            ),
            dtype=np.float32,
        ).reshape(-1)


class WhisperSpeechToText:
    """Native VAD only (no Moonshine ASR weights) + Whisper CPU INT8 decoding."""

    def __init__(self, config, callback, progress=lambda *args: None):
        from faster_whisper import WhisperModel
        from moonshine_voice import ModelArch, Transcriber

        os.environ.setdefault("MOONSHINE_ORT_SINGLE_THREAD", "1")
        self.callback = callback
        model = config.stt_model.removeprefix("whisper_").replace("_en", ".en")
        self.language = "en" if model.endswith(".en") else None
        progress(0, f"Loading Whisper {model} INT8; first use downloads weights")
        self.model = WhisperModel(
            model, device="cpu", compute_type="int8", cpu_threads=config.stt_threads, num_workers=1
        )
        self.keyterms = " ".join(config.stt_keyterms) or None
        self.transcriber = Transcriber(
            ".",
            ModelArch.TINY_STREAMING,
            options={
                "skip_transcription": "true",
                "return_audio_data": "true",
                "word_timestamps": "false",
                "vad_max_segment_duration": 20.0,
            },
        )
        self.stream = None
        self.reset()

    def _event(self, event):
        from moonshine_voice import LineCompleted, LineStarted

        if isinstance(event, LineStarted):
            self.callback("speech", "")
        elif isinstance(event, LineCompleted):
            audio = np.asarray(event.line.audio_data or [], dtype=np.float32)
            if len(audio) < 1600 or float(np.sqrt(np.mean(audio**2))) < 0.001:
                self.callback("transcript", "")
                return
            self.callback("partial", "Transcribing locally…")
            language = self.language
            if language is None:
                # This companion only needs English/Indonesian. Short clips can
                # otherwise be misidentified as an unrelated language/script.
                _, _, probabilities = self.model.detect_language(audio=audio)
                candidates = [(prob, lang) for lang, prob in probabilities if lang in {"en", "id"}]
                language = max(candidates, default=(0, "en"))[1]
            segments, _ = self.model.transcribe(
                audio,
                language=language,
                task="transcribe",
                multilingual=False,
                beam_size=5,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=False,
                hotwords=self.keyterms,
                max_new_tokens=160,
            )
            text = " ".join(
                segment.text.strip() for segment in segments if segment.no_speech_prob < 0.8
            ).strip()
            self.callback("transcript", text)

    def reset(self):
        if self.stream:
            self.stream.remove_all_listeners()
            self.stream.close()
        self.stream = self.transcriber.create_stream(update_interval=0.1)
        self.stream.add_listener(self._event)
        self.stream.start()

    def feed(self, audio, rate):
        self.stream.add_audio(audio, rate)

    def close(self):
        if self.stream:
            self.stream.remove_all_listeners()
            self.stream.close()
        self.transcriber.close()
        self.model = None


def _kokoro_file(name):
    from uuid import uuid4

    import httpx

    directory = Path.home() / ".cache/2tor/kokoro-v1.0"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    if target.is_file():
        return str(target)
    partial = directory / (name + "." + uuid4().hex + ".partial")
    url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/" + name
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
            response.raise_for_status()
            with partial.open("wb") as file:
                for chunk in response.iter_bytes():
                    file.write(chunk)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return str(target)


class KokoroSpeech:
    def __init__(self, config):
        import espeakng_loader
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        from kokoro_onnx.config import EspeakConfig

        options = ort.SessionOptions()
        options.intra_op_num_threads = config.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.enable_cpu_mem_arena = False
        session = ort.InferenceSession(
            _kokoro_file("kokoro-v1.0.int8.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.model = Kokoro.from_session(
            session,
            _kokoro_file("voices-v1.0.bin"),
            espeak_config=EspeakConfig(
                lib_path=espeakng_loader.get_library_path(),
                data_path=espeakng_loader.get_data_path(),
            ),
        )
        self.config = config
        self.generate("Hello.")

    def generate(self, text, voice=None, speed=None, cancel=None):
        voice_id = KOKORO_VOICES[voice or self.config.voice]
        audio, rate = self.model.create(
            text,
            voice=voice_id,
            speed=speed or self.config.speed,
            lang="en-gb" if voice_id.startswith("b") else "en-us",
            sentence_pause=0.12,
            clause_pause=0.06,
        )
        if rate != 24000:
            raise RuntimeError("Unexpected Kokoro sample rate")
        return np.asarray(audio, dtype=np.float32).reshape(-1)


def decode_edge_audio(data):
    """Decode MP3 in memory to the existing mono float32 24kHz playback format."""
    import io

    import av

    chunks = []
    with av.open(io.BytesIO(data), format="mp3") as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=24000)
        for frame in container.decode(audio=0):
            for output in resampler.resample(frame):
                chunks.append(output.to_ndarray().reshape(-1))
        for output in resampler.resample(None):
            chunks.append(output.to_ndarray().reshape(-1))
    if not chunks:
        raise RuntimeError("Edge TTS returned no decodable audio")
    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    if len(audio) > 24000 * 60 or not np.isfinite(audio).all():
        raise RuntimeError("Edge TTS audio exceeded the supported preview/chunk limits")
    return audio


class EdgeSpeech:
    """Online Microsoft speech service; cancellation closes the network request."""

    def __init__(self, config):
        self.config = config  # No model downloads, network warm-up, or API key.

    def generate(self, text, voice=None, speed=None, cancel=None):
        import asyncio

        import edge_tts

        voice = voice or self.config.voice
        if voice not in EDGE_VOICES:
            raise ValueError("Unknown Edge TTS voice")
        if cancel is not None and cancel.is_set():
            return np.empty(0, dtype=np.float32)
        rate = f"{round(((speed or self.config.speed) - 1) * 100):+d}%"

        async def fetch():
            data = bytearray()
            communication = edge_tts.Communicate(
                text, EDGE_VOICES[voice], rate=rate, connect_timeout=10, receive_timeout=15
            )
            async for chunk in communication.stream():
                if chunk["type"] == "audio":
                    data.extend(chunk["data"])
                    if len(data) > 8 * 1024 * 1024:
                        raise RuntimeError("Edge TTS audio response exceeded 8 MB")
            if not data:
                raise RuntimeError("Edge TTS returned no audio")
            return bytes(data)

        async def run():
            task = asyncio.create_task(fetch())
            try:
                while not task.done():
                    await asyncio.wait({task}, timeout=0.05)
                    if cancel is not None and cancel.is_set():
                        task.cancel()
                return await task
            finally:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        try:
            data = asyncio.run(asyncio.wait_for(run(), timeout=25))
        except asyncio.CancelledError:
            return np.empty(0, dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(
                "Edge TTS unavailable. Check internet or select Kokoro/Kitten."
            ) from exc
        if cancel is not None and cancel.is_set():
            return np.empty(0, dtype=np.float32)
        return decode_edge_audio(data)
