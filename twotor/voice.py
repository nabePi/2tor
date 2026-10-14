"""Local CPU engines. No torch imports, speaker identification, or saved audio."""

import json
import os
from pathlib import Path

import numpy as np


class SpeechToText:
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
        self.stream = None
        self.completed = False
        self.reset()

    def reset(self):
        if self.stream is not None:
            self.stream.remove_all_listeners()
            self.stream.close()
        self.completed = False
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
            self.completed = True
            self.callback("transcript", event.line.text)

    def feed(self, audio, rate):
        self.stream.add_audio(audio, rate)
        if self.completed:
            self.reset()  # Release the previous turn's transcript/audio state.

    def close(self):
        if self.stream:
            self.stream.remove_all_listeners()
            self.stream.close()
        self.transcriber.close()


class TextToSpeech:
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

    def generate(self, text):
        return np.asarray(
            self.model.generate(
                text, voice=self.config.voice, speed=self.config.speed, clean_text=True
            ),
            dtype=np.float32,
        ).reshape(-1)
