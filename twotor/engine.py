import queue
import threading
import time

import psutil
import sounddevice as sd

from .config import VOICES
from .conversation import IdlePolicy, SentenceBuffer
from .llm import DeepSeek
from .voice import SpeechToText, TextToSpeech


def memory_mb():
    process = psutil.Process()
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total / 1024**2


class Engine:
    """One capture/STT owner; one bounded, cancellable response pipeline."""

    def __init__(self, config):
        self.config = config
        self.events = queue.Queue(maxsize=256)
        self.commands = queue.Queue(maxsize=32)
        self.results = queue.Queue(maxsize=1)
        self.audio = queue.Queue(maxsize=24)
        self.listen = threading.Event()
        self.quit = threading.Event()
        self.cancel = threading.Event()
        self.active = False
        self.focus = False
        self.busy = False
        self.speaking = False
        self.generation = 0
        self.work_generation = None
        self.capture = None
        self.stt = None
        self.tts = None
        self.voice = config.voice
        self.history = []
        self.policy = IdlePolicy(config.proactive_seconds, time.monotonic())
        self.thread = threading.Thread(target=self.run, name="2tor-audio", daemon=True)
        self.response_thread = None
        self.pending = ""
        self.pending_at = 0.0
        self.speech_at = 0.0
        self.session = None
        self.thread.start()

    def emit(self, name, data="", generation=None):
        item = (name, data, generation)
        try:
            self.events.put_nowait(item)
        except queue.Full:
            # A hidden/frozen UI must not grow memory without bounds.
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            self.events.put_nowait(item)

    def command(self, name, data=None):
        # Stop audio immediately, even while inference is running.
        if name in {"stop", "pause", "quit"} or (name == "focus" and data):
            self.cancel.set()
            self.listen.clear()
            sd.stop()
        if name == "quit":
            self.quit.set()
        try:
            self.commands.put_nowait((name, data))
        except queue.Full:
            self.emit("error", "Too many actions; please wait briefly.")

    def _capture(self, audio, frames, timing, status):
        if not self.listen.is_set():
            return
        if status:
            self.emit("notice", "Audio capture overloaded. Try TTS_THREADS=1.")
        try:
            self.audio.put_nowait(audio[:, 0].copy())
        except queue.Full:
            # Drop the newest block; never block PortAudio's callback.
            self.emit("notice", "Audio backlog: a block was dropped. Try a smaller STT model.")

    def _transcript(self, name, data):
        if not self.listen.is_set():
            return
        now = time.monotonic()
        if name in {"speech", "partial"}:
            self.speaking = True
            self.speech_at = now
            self.policy.activity(now)
        if name == "partial":
            self.emit("partial", data)
        elif name == "transcript":
            self.speaking = False
            if data.strip():
                self.pending = (self.pending + " " + data.strip()).strip()[:4000]
                self.pending_at = now
                self.policy.activity(now)
            self.emit("partial", "")

    def _drain_audio(self):
        while True:
            try:
                self.audio.get_nowait()
            except queue.Empty:
                break

    def _open_capture(self):
        if self.capture:
            return
        info = sd.query_devices(self.config.input_device, "input")
        self.rate = int(info["default_samplerate"])
        self.capture = sd.InputStream(
            device=self.config.input_device,
            samplerate=self.rate,
            channels=1,
            blocksize=max(1, self.rate // 10),
            dtype="float32",
            callback=self._capture,
        )
        self.capture.start()

    def _pause(self):
        self.active = False
        self.listen.clear()
        self.cancel.set()
        sd.stop()
        self.generation += 1
        # Do not reuse a TTS instance while a cancelled synthesis still runs.
        self.busy = bool(self.response_thread and self.response_thread.is_alive())
        if self.capture:
            self.capture.close()
            self.capture = None
        self.pending = ""
        self.speaking = False
        self._drain_audio()
        if self.stt:
            self.stt.reset()
        self.emit("partial", "")
        self.emit("state", "Paused")

    def _start(self):
        if not self.config.api_key:
            raise RuntimeError("Fill DEEPSEEK_API_KEY in .env, then restart 2tor.")
        if self.response_thread and self.response_thread.is_alive():
            raise RuntimeError("Previous reply is stopping. Wait briefly, then start again.")
        self.busy = False
        self.emit("state", "Loading local voices…")
        if self.stt is None:
            self.stt = SpeechToText(
                self.config,
                self._transcript,
                lambda fraction, file: self.emit(
                    "notice", f"STT download: {fraction:.0%} · {file}"
                ),
            )
        if self.tts is None:
            self.emit("notice", "Loading KittenTTS Nano INT8; first run downloads the model.")
            self.tts = TextToSpeech(self.config)
        # Honor pause/quit sent during downloads; process it in the command loop.
        if not self.commands.empty() or self.quit.is_set():
            self.emit("state", "Ready")
            return
        if memory_mb() > self.config.memory_limit_mb:
            raise RuntimeError("Memory target exceeded while loading. Close other 2tor instances.")
        self._open_capture()
        self.active = True
        self.policy.activity(time.monotonic())
        if self.focus:
            self.emit("state", "Focus")
        else:
            self.listen.set()
            self.emit("state", "Listening")
            if not self.history:
                self._reply(
                    "Start the session: greet me briefly and open an easy English topic.",
                    proactive=True,
                )

    def _reply(self, instruction=None, proactive=False):
        if self.busy or not self.active or self.focus:
            return
        self.busy = True
        self.listen.clear()
        self._drain_audio()
        self.speaking = False
        self.pending = ""
        self.stt.reset()
        self.generation += 1
        generation = self.generation
        self.work_generation = generation
        self.cancel = threading.Event()
        cancel = self.cancel
        history = list(self.history)
        voice = self.voice  # Keep one voice throughout this reply.
        if proactive:
            self.policy.prompted(time.monotonic())
        self.emit("state", "Thinking")

        def pipeline():
            sentences = queue.Queue(maxsize=3)
            result = ""
            errors = []
            tts = self.tts

            def say():
                try:
                    while not cancel.is_set():
                        try:
                            text = sentences.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        if text is None:
                            break
                        audio = tts.generate(text, voice=voice)
                        if cancel.is_set():
                            break
                        self.emit("state", "Speaking", generation)
                        sd.play(audio, 24000, device=self.config.output_device)
                        while sd.get_stream().active:
                            if cancel.wait(0.04):
                                sd.stop()
                                break
                        if not cancel.is_set():
                            self.emit("state", "Thinking", generation)
                except Exception as exc:
                    errors.append(f"Voice playback failed: {exc}")
                    cancel.set()

            speaker = threading.Thread(target=say, name="2tor-tts", daemon=True)
            speaker.start()

            def put(text):
                while not cancel.is_set():
                    try:
                        sentences.put(text, timeout=0.1)
                        return
                    except queue.Full:
                        pass

            try:
                splitter = SentenceBuffer()
                for delta in DeepSeek(self.config).stream(history, instruction, cancel):
                    if cancel.is_set():
                        break
                    result += delta
                    self.emit("reply", result, generation)
                    for sentence in splitter.feed(delta):
                        put(sentence)
                if not cancel.is_set():
                    for sentence in splitter.feed("", final=True):
                        put(sentence)
                    put(None)
            except Exception as exc:
                errors.append(str(exc))
                cancel.set()
                sd.stop()
            finally:
                speaker.join()
                # Let room echo decay before reopening capture.
                if not cancel.is_set():
                    cancel.wait(0.35)
                data = {"text": result, "errors": errors, "cancelled": cancel.is_set()}
                self.results.put((data, generation))
                self.emit("done", data, generation)

        self.response_thread = threading.Thread(target=pipeline, name="2tor-reply", daemon=True)
        self.response_thread.start()

    def _preview(self, voice):
        if voice not in VOICES:
            self.emit("error", "Unknown TTS voice")
            return
        if self.busy:
            self.emit("notice", "Stop or finish the current reply before previewing a voice.")
            return
        if self.focus:
            self.emit("notice", "Turn off Focus / mute to hear a voice preview.")
            return
        self.voice = voice
        self.busy = True
        self.listen.clear()
        self._drain_audio()
        self.pending = ""
        self.speaking = False
        if self.stt:
            self.stt.reset()
        self.generation += 1
        generation = self.generation
        self.work_generation = generation
        self.cancel = threading.Event()
        cancel = self.cancel
        self.emit("state", "Loading voice sample…", generation)

        def preview():
            errors = []
            try:
                if self.tts is None:
                    self.emit("notice", "First preview downloads the local TTS model.", generation)
                    self.tts = TextToSpeech(self.config)
                if cancel.is_set():
                    return
                if memory_mb() > self.config.memory_limit_mb:
                    raise RuntimeError("Memory target exceeded; restart 2tor before previewing.")
                text = (
                    f"Hi, I'm {voice}, your English conversation partner. "
                    "How is your day going? Let's practice a little English together."
                )
                audio = self.tts.generate(text, voice=voice)
                if cancel.is_set():
                    return
                self.emit("state", f"Previewing {voice}", generation)
                sd.play(audio, 24000, device=self.config.output_device)
                while sd.get_stream().active:
                    if cancel.wait(0.04):
                        sd.stop()
                        break
                if not cancel.is_set():
                    cancel.wait(0.35)
            except Exception as exc:
                errors.append(f"Voice preview failed: {exc}")
                cancel.set()
                sd.stop()
            finally:
                data = {"text": "", "errors": errors, "cancelled": cancel.is_set(), "preview": True}
                self.results.put((data, generation))
                self.emit("done", data, generation)

        self.response_thread = threading.Thread(target=preview, name="2tor-preview", daemon=True)
        self.response_thread.start()

    def finish(self, data, generation):
        # Called on the engine owner thread, independently of UI polling.
        if generation != self.generation:
            if generation == self.work_generation:
                self.busy = False
                self.emit(
                    "state", "Focus" if self.focus else "Listening" if self.active else "Paused"
                )
            return
        self.busy = False
        if data["text"] and not data.get("preview"):
            text = data["text"]
            if data["cancelled"]:
                text += " [interrupted]"
            self.history.append({"role": "assistant", "content": text})
            self.history = self.history[-20:]
            self._save("assistant", text)
        for error in data["errors"]:
            self.emit("error", error)
        self.policy.last_activity = time.monotonic()
        self._drain_audio()
        if self.stt:
            self.stt.reset()
        if self.active and not self.focus:
            self.listen.set()
        self.emit("state", "Focus" if self.focus else "Listening" if self.active else "Ready")

    def _save(self, role, text):
        if not self.config.save_history:
            return
        import sqlite3
        from pathlib import Path
        from uuid import uuid4

        if self.session is None:
            self.session = str(uuid4())
        directory = Path.home() / ".local/share/2tor"
        directory.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(directory / "history.sqlite3") as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS messages "
                "(session TEXT, time REAL, role TEXT, text TEXT)"
            )
            db.execute(
                "INSERT INTO messages VALUES (?,?,?,?)", (self.session, time.time(), role, text)
            )

    def run(self):
        next_memory = 0
        next_idle_reset = time.monotonic() + 30
        try:
            while not self.quit.is_set():
                try:
                    self.finish(*self.results.get_nowait())
                except queue.Empty:
                    pass
                except Exception as exc:
                    self._pause()
                    self.emit("error", str(exc))
                try:
                    name, data = self.commands.get_nowait()
                    if name == "start":
                        self._start()
                    elif name == "voice":
                        if data in VOICES:
                            self.voice = data
                            self.emit("notice", f"{data} will speak the next reply.")
                        else:
                            self.emit("error", "Unknown TTS voice")
                    elif name == "preview":
                        self._preview(data)
                    elif name == "pause":
                        self._pause()
                    elif name == "stop":
                        self.pending = ""
                        self.speaking = False
                        if not self.busy and self.active and not self.focus:
                            self.stt.reset()
                            self._drain_audio()
                            self.listen.set()
                            self.emit("state", "Listening")
                    elif name == "focus":
                        self.focus = bool(data)
                        self.policy.last_activity = time.monotonic()
                        self.pending = ""
                        self.speaking = False
                        if self.active and not self.busy:
                            self.stt.reset()
                            self._drain_audio()
                            if not self.focus:
                                self.listen.set()
                            self.emit("state", "Focus" if self.focus else "Listening")
                    elif name == "text":
                        if self.active and not self.busy and not self.focus and data.strip():
                            text = data.strip()[:4000]
                            self.history.append({"role": "user", "content": text})
                            self.history = self.history[-20:]
                            self._save("user", text)
                            self.emit("user", text)
                            self.policy.activity(time.monotonic())
                            self._reply()
                        else:
                            self.emit(
                                "notice", "Start a session and finish the current reply first."
                            )
                    elif name == "topic":
                        self._reply("Open a fresh, easy topic. Ask me one short question.", True)
                    elif name == "done":
                        self.finish(*data)
                    elif name == "interval":
                        self.policy.seconds = int(data)
                        self.policy.last_activity = time.monotonic()
                    elif name == "clear":
                        if self.busy:
                            self.emit("notice", "Stop the current reply before clearing the chat.")
                        else:
                            self.history.clear()
                            self.session = None
                            self.emit("cleared")
                except queue.Empty:
                    pass
                except Exception as exc:
                    self._pause()
                    self.emit("error", str(exc))

                try:
                    audio = self.audio.get(timeout=0.05)
                    if self.listen.is_set() and self.stt:
                        self.stt.feed(audio, self.rate)
                except queue.Empty:
                    pass
                except Exception as exc:
                    self._pause()
                    self.emit("error", f"Microphone/STT failed: {exc}")
                now = time.monotonic()
                if self.speaking and now - self.speech_at > 30:
                    self.speaking = False
                if (
                    now >= next_idle_reset
                    and self.listen.is_set()
                    and not self.speaking
                    and not self.pending
                    and self.stt
                ):
                    self.stt.reset()
                    self._drain_audio()
                    next_idle_reset = now + 30
                if (
                    self.pending
                    and not self.speaking
                    and not self.busy
                    and self.active
                    and not self.focus
                    and now - self.pending_at > 0.7
                ):
                    text = self.pending
                    self.commands.put_nowait(("text", text))
                    self.pending = ""
                if self.policy.due(
                    now,
                    active=self.active,
                    busy=self.busy,
                    focus=self.focus,
                    speaking=self.speaking or bool(self.pending),
                ):
                    self._reply(
                        "I have been quiet. Gently check in or open a new topic. "
                        "Do not assume I am ignoring you; I may be working.",
                        True,
                    )
                if now >= next_memory:
                    next_memory = now + 5
                    ram = memory_mb()
                    self.emit("memory", ram)
                    if ram > self.config.memory_limit_mb and self.active:
                        self._pause()
                        self.emit(
                            "error",
                            "Memory target exceeded; session paused. "
                            "Restart 2tor to release model memory.",
                        )
        finally:
            self.listen.clear()
            self.cancel.set()
            sd.stop()
            if self.capture:
                self.capture.close()
            if self.stt:
                self.stt.close()
