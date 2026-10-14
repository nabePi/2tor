import argparse
import html
import os
import queue
import sys

from .config import Config, voices_for_model


def main():
    parser = argparse.ArgumentParser(description="2tor · English while you work")
    parser.add_argument("--env", help="Path to your .env file")
    parser.add_argument("--devices", action="store_true", help="List audio input/output devices")
    parser.add_argument("--smoke-test", action="store_true", help="Open UI briefly, no mic/API")
    parser.add_argument(
        "--voice-test",
        action="store_true",
        help="Test configured voices, no mic/DeepSeek (Edge is online)",
    )
    args = parser.parse_args()
    if args.devices:
        import sounddevice as sd

        print(sd.query_devices())
        return
    if args.voice_test:
        from .benchmark import main as benchmark

        benchmark(["--turns", "3"] + (["--env", args.env] if args.env else []))
        return
    try:
        config = Config.load(args.env)
    except ValueError as exc:
        parser.error(str(exc))
    # Set before importing numerical libraries. TTS also sets ORT SessionOptions.
    os.environ.setdefault("OMP_NUM_THREADS", str(config.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from PySide6.QtCore import QEvent, Qt, QTimer
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )

    from .engine import Engine

    class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("2tor · English while you work")
            self.resize(500, 800)
            self.setMinimumSize(390, 480)
            self.engine = Engine(config)
            self.running = False
            self.generation = None
            self.messages = []
            self.current_reply = ""
            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)
            layout.setContentsMargins(24, 24, 24, 22)
            layout.setSpacing(10)
            title = QLabel("2tor")
            title.setObjectName("title")
            layout.addWidget(title)
            subtitle = QLabel("A little English. A little company.")
            subtitle.setObjectName("muted")
            layout.addWidget(subtitle)
            top = QHBoxLayout()
            self.status = QLabel("●  Ready")
            self.status.setObjectName("status")
            self.ram = QLabel("— MB")
            self.ram.setObjectName("muted")
            top.addWidget(self.status)
            top.addStretch()
            top.addWidget(self.ram)
            layout.addLayout(top)
            self.chat = QTextBrowser()
            self.chat.setOpenExternalLinks(False)
            layout.addWidget(self.chat, 1)
            self.partial = QLabel("Start a session when you’re ready.")
            self.partial.setWordWrap(True)
            self.partial.setObjectName("muted")
            layout.addWidget(self.partial)
            row = QHBoxLayout()
            self.start = QPushButton("Start session")
            self.start.setObjectName("primary")
            self.start.clicked.connect(self.toggle)
            stop = QPushButton("Stop reply")
            stop.clicked.connect(lambda: self.engine.command("stop"))
            row.addWidget(self.start)
            row.addWidget(stop)
            layout.addLayout(row)
            controls = QHBoxLayout()
            self.focus = QCheckBox("Focus / mute")
            self.focus.toggled.connect(lambda value: self.engine.command("focus", value))
            pin = QCheckBox("Keep on top")
            pin.toggled.connect(self.pin)
            controls.addWidget(self.focus)
            controls.addStretch()
            controls.addWidget(pin)
            layout.addLayout(controls)
            voice_row = QHBoxLayout()
            voice_row.addWidget(QLabel("Voice"))
            self.tts_engine = QComboBox()
            self.tts_engine.addItem("Edge (online)", "edge_tts")
            self.tts_engine.addItem("Kokoro", "kokoro_int8")
            self.tts_engine.addItem("Kitten", "KittenML/kitten-tts-nano-0.8-int8")
            if self.tts_engine.findData(config.tts_model) < 0:
                self.tts_engine.addItem("Kitten custom", config.tts_model)
            self.tts_engine.setCurrentIndex(self.tts_engine.findData(config.tts_model))
            self.tts_engine.currentIndexChanged.connect(
                lambda _: self.engine.command("tts_model", self.tts_engine.currentData())
            )
            voice_row.addWidget(self.tts_engine)
            self.voice = QComboBox()
            self.voice.addItems(voices_for_model(config.tts_model))
            self.voice.setCurrentText(config.voice)
            self.voice.currentTextChanged.connect(lambda name: self.engine.command("voice", name))
            self.preview = QPushButton("Listen to sample")
            self.preview.setToolTip("Hear this voice without a DeepSeek key. Edge uses internet.")
            self.preview.clicked.connect(
                lambda: self.engine.command("preview", self.voice.currentText())
            )
            voice_row.addWidget(self.voice, 1)
            voice_row.addWidget(self.preview)
            layout.addLayout(voice_row)
            speed_row = QHBoxLayout()
            speed_row.addWidget(QLabel("Speaking speed"))
            speed = self.speed = QDoubleSpinBox()
            speed.setRange(0.7, 1.3)
            speed.setSingleStep(0.05)
            speed.setValue(config.speed)
            speed.setSuffix(" ×")
            speed.valueChanged.connect(lambda value: self.engine.command("speed", value))
            speed_row.addWidget(speed, 1)
            layout.addLayout(speed_row)
            recognition_row = QHBoxLayout()
            recognition_row.addWidget(QLabel("Recognition"))
            self.recognition = QComboBox()
            self.recognition.addItem("Whisper Small · English + Indonesia", "whisper_small")
            self.recognition.addItem("Whisper Base · bilingual, faster", "whisper_base")
            self.recognition.addItem("Whisper Small · English only", "whisper_small_en")
            self.recognition.addItem("Whisper Base · English only", "whisper_base_en")
            self.recognition.addItem("Moonshine Small · English only", "small_streaming")
            self.recognition.addItem("Moonshine Tiny · English only", "tiny_streaming")
            self.recognition.setCurrentIndex(self.recognition.findData(config.stt_model))
            self.recognition.setToolTip("Pause the session before changing recognition")
            self.recognition.currentIndexChanged.connect(
                lambda _: self.engine.command("stt_model", self.recognition.currentData())
            )
            recognition_row.addWidget(self.recognition, 1)
            layout.addLayout(recognition_row)
            mic_row = QHBoxLayout()
            mic_row.addWidget(QLabel("Microphone"))
            self.microphone = QComboBox()
            self.microphone.addItem("System default", None)
            import sounddevice as sd

            try:
                for index, info in enumerate(sd.query_devices()):
                    if info["max_input_channels"]:
                        self.microphone.addItem(info["name"], index)
            except sd.PortAudioError:
                self.notice_text = "No microphone devices found. Check audio settings."
            index = self.microphone.findData(config.input_device)
            if index < 0:
                self.microphone.addItem(str(config.input_device), config.input_device)
                index = self.microphone.count() - 1
            self.microphone.setCurrentIndex(index)
            self.microphone.currentIndexChanged.connect(
                lambda _: self.engine.command("input_device", self.microphone.currentData())
            )
            mic_row.addWidget(self.microphone, 1)
            self.mic_test = QPushButton("Test mic")
            self.mic_test.clicked.connect(
                lambda: self.engine.command("pause" if self.engine.testing else "mic_test")
            )
            mic_row.addWidget(self.mic_test)
            layout.addLayout(mic_row)
            level_row = QHBoxLayout()
            self.level = QProgressBar()
            self.level.setRange(0, 60)
            self.level.setValue(0)
            self.level.setTextVisible(False)
            self.level.setMaximumHeight(7)
            self.level_label = QLabel("Mic level")
            self.level_label.setObjectName("muted")
            level_row.addWidget(self.level, 1)
            level_row.addWidget(self.level_label)
            layout.addLayout(level_row)
            proactive = QHBoxLayout()
            wait_row = QHBoxLayout()
            wait_row.addWidget(QLabel("Thinking pause"))
            self.end_pause = QDoubleSpinBox()
            self.end_pause.setRange(0.5, 15)
            self.end_pause.setSingleStep(0.5)
            self.end_pause.setDecimals(1)
            self.end_pause.setSuffix(" seconds")
            self.end_pause.setValue(config.stt_end_pause)
            self.end_pause.setToolTip("Wait after speech; hmm / hemm / enggg restart this timer.")
            self.end_pause.valueChanged.connect(
                lambda value: self.engine.command("end_pause", value)
            )
            wait_row.addWidget(self.end_pause, 1)
            layout.addLayout(wait_row)
            proactive.addWidget(QLabel("Quiet check-in"))
            interval = QComboBox()
            for label, seconds in [
                ("90 seconds", 90),
                ("3 minutes", 180),
                ("5 minutes", 300),
                ("10 minutes", 600),
            ]:
                interval.addItem(label, seconds)
            if interval.findData(config.proactive_seconds) == -1:
                interval.addItem(f"{config.proactive_seconds} seconds", config.proactive_seconds)
            interval.setCurrentIndex(interval.findData(config.proactive_seconds))
            interval.currentIndexChanged.connect(
                lambda _: self.engine.command("interval", interval.currentData())
            )
            proactive.addWidget(interval)
            topic = QPushButton("New topic")
            topic.clicked.connect(lambda: self.engine.command("topic"))
            proactive.addWidget(topic)
            layout.addLayout(proactive)
            textrow = QHBoxLayout()
            self.input = QLineEdit()
            self.input.setPlaceholderText("Type in English or bahasa Indonesia…")
            self.input.setMaxLength(4000)
            self.input.returnPressed.connect(self.send)
            send = QPushButton("Send")
            send.clicked.connect(self.send)
            textrow.addWidget(self.input, 1)
            textrow.addWidget(send)
            layout.addLayout(textrow)
            footer = QHBoxLayout()
            clear = QPushButton("Clear chat")
            clear.clicked.connect(lambda: self.engine.command("clear"))
            footer.addWidget(clear)
            footer.addStretch()
            label = self.voice_status = QLabel(
                "Voice online · AI online"
                if config.tts_model == "edge_tts"
                else "Voice local · AI online"
            )
            label.setObjectName("muted")
            footer.addWidget(label)
            layout.addLayout(footer)
            self.notice = QLabel("Headphones help keep the conversation clear.")
            self.notice.setWordWrap(True)
            self.notice.setObjectName("muted")
            layout.addWidget(self.notice)
            self.setStyleSheet("""
                QWidget { background: #171c1b; color: #edf0e9;
                    font-family: 'Helvetica Neue', 'DejaVu Sans'; font-size: 13px; }
                QLabel#title { font-size: 39px; font-weight: 600; color: #c5e4ae; }
                QLabel#muted { color: #97a39a; font-size: 12px; }
                QLabel#status { color: #c5e4ae; padding: 8px 0; }
                QTextBrowser { background: #202724; border: 1px solid #354039;
                    border-radius: 10px; padding: 13px; font-size: 15px; }
                QPushButton { background: #2b352f; border: 1px solid #465349;
                    border-radius: 7px; padding: 10px 12px; }
                QPushButton:hover { background: #39463c; }
                QPushButton:disabled { color: #778078; }
                QPushButton#primary { background: #c5e4ae; color: #1b291c; font-weight: 600; }
                QLineEdit, QComboBox, QDoubleSpinBox { background: #202724; border: 1px solid #465349;
                    border-radius: 6px; padding: 9px; }
                QCheckBox { spacing: 7px; }
                QProgressBar { background: #2b352f; border: none; border-radius: 3px; }
                QProgressBar::chunk { background: #c5e4ae; border-radius: 3px; }
            """)
            self.render()
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.poll)
            self.timer.start(60)

        def toggle(self):
            self.engine.command("pause" if self.running else "start")
            self.start.setEnabled(False)

        def pin(self, checked):
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, checked)
            self.show()

        def send(self):
            if self.input.text().strip():
                self.engine.command("text", self.input.text())
                self.input.clear()

        def render(self):
            blocks = []
            for role, text in self.messages[-80:]:
                color = "#c5e4ae" if role == "2tor" else "#97a39a"
                blocks.append(
                    f'<p style="color:{color}; font-size:11px">{role}</p><p>{html.escape(text)}</p>'
                )
            if self.current_reply:
                blocks.append(
                    '<p style="color:#c5e4ae; font-size:11px">2tor</p>'
                    f"<p>{html.escape(self.current_reply)}</p>"
                )
            if not blocks:
                blocks.append(
                    "<p>Tell me about your day, think out loud, or practice a small "
                    "conversation while you work.</p><p style='color:#97a39a'>"
                    "I can open a topic when things get quiet.</p>"
                )
            self.chat.setHtml("".join(blocks))
            self.chat.verticalScrollBar().setValue(self.chat.verticalScrollBar().maximum())

        def poll(self):
            for _ in range(100):
                try:
                    name, data, generation = self.engine.events.get_nowait()
                except queue.Empty:
                    break
                if name == "done":
                    if data.get("preview"):
                        continue
                    if generation == self.generation:
                        if data["text"]:
                            text = data["text"] + (" [interrupted]" if data["cancelled"] else "")
                            self.messages.append(("2tor", text))
                        self.current_reply = ""
                        self.render()
                    continue
                if generation is not None and generation != self.engine.generation:
                    continue
                if name == "reply":
                    self.generation = generation
                    self.current_reply = data
                    self.render()
                elif name == "user":
                    self.messages.append(("You", data))
                    self.messages = self.messages[-80:]
                    self.partial.setText("")
                    self.render()
                elif name == "partial":
                    self.partial.setText(data)
                elif name == "state":
                    self.status.setText("●  " + data)
                    self.running = self.engine.active
                    self.start.setText("Pause session" if self.running else "Start session")
                    self.start.setEnabled(
                        self.running or (not self.engine.busy and not data.startswith("Loading"))
                    )
                elif name in {"error", "notice"}:
                    self.notice.setText(data)
                elif name == "memory":
                    self.ram.setText(f"{data:.0f} MB")
                elif name == "tts_changed":
                    self.voice_status.setText(
                        "Voice online · AI online"
                        if self.engine.tts_model == "edge_tts"
                        else "Voice local · AI online"
                    )
                    self.voice.blockSignals(True)
                    self.voice.clear()
                    self.voice.addItems(data["voices"])
                    self.voice.setCurrentText(data["voice"])
                    self.voice.blockSignals(False)
                    self.speed.setValue(data["speed"])
                elif name == "level":
                    db, peak = data
                    self.level.setValue(int(max(0, min(60, db + 60))))
                    self.level_label.setText(
                        "Clipping"
                        if peak >= 0.98
                        else "Low / quiet"
                        if db < -45
                        else f"{db:.0f} dB"
                    )
                elif name == "cleared":
                    self.messages.clear()
                    self.current_reply = ""
                    self.render()
            self.preview.setEnabled(not self.engine.busy and not self.focus.isChecked())
            self.recognition.setEnabled(not self.engine.active and not self.engine.busy)
            self.tts_engine.setEnabled(not self.engine.active and not self.engine.busy)
            self.microphone.setEnabled(not self.engine.active and not self.engine.busy)
            self.mic_test.setEnabled(
                self.engine.testing or (not self.engine.active and not self.engine.busy)
            )
            self.mic_test.setText("Stop test" if self.engine.testing else "Test mic")

        def closeEvent(self, event):
            self.timer.stop()
            self.engine.command("quit")
            self.engine.thread.join(timeout=2)
            if self.engine.response_thread:
                self.engine.response_thread.join(timeout=2)
            event.accept()

    app = QApplication(sys.argv[:1])
    app.setApplicationName("2tor")
    window = Window()
    window.show()
    if args.smoke_test:
        os.makedirs("artifacts", exist_ok=True)
        QTimer.singleShot(400, lambda: window.grab().save("artifacts/ui-preview.png"))
        QTimer.singleShot(800, window.close)
    exit_code = app.exec()
    # Dispose of widgets while QApplication is still alive, before Python
    # teardown and native-library finalizers in the packaged app.
    window.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    sys.exit(exit_code)
