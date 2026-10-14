# 2tor

2tor adalah aplikasi desktop untuk menemani kamu bekerja sambil melatih percakapan bahasa Inggris. Kamu berbicara lewat mikrofon, AI menjawab dengan suara, dan AI bisa membuka topik saat kamu diam.

Alur: **mikrofon → STT lokal → DeepSeek API → TTS lokal → speaker/headphone**.

Aplikasi menggunakan Python + PySide6 Qt Widgets, STT Moonshine Small Streaming English (dengan opsi Tiny), dan TTS KittenTTS Nano INT8 melalui ONNX Runtime. GPU tidak diperlukan. Model suara diunduh saat pertama dipakai lalu digunakan dari cache lokal. Percakapan dengan DeepSeek tetap membutuhkan internet dan API key.

## Perangkat yang ditargetkan

| Sistem | Perangkat |
|---|---|
| macOS Apple Silicon | Mac mini M4 Pro, RAM 24 GB |
| Ubuntu 26.04 LTS, x86_64 | Lenovo ThinkPad X390, Intel Core i5 generasi 8 vPro, RAM 16 GB |

UI, mesin suara lokal, dan paket `.app` sudah diuji di macOS. Ubuntu disiapkan melalui source dan skrip build; pengujian langsung pada X390 masih diperlukan.

## Menjalankan di macOS

### 1. Instal uv

Jika Homebrew sudah tersedia:

```bash
brew install uv
```

Atau gunakan installer resmi tanpa Homebrew:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Buka ulang Terminal setelah instalasi, lalu periksa:

```bash
uv --version
```

Lihat [panduan resmi uv](https://docs.astral.sh/uv/getting-started/installation/) jika perintah belum ditemukan. Setup akan menyiapkan Python 3.11 melalui uv, jadi Python tidak perlu diinstal secara terpisah.

### 2. Siapkan aplikasi

Untuk lokasi proyek pada Mac ini:

```bash
cd /Users/wahyusaid/projects/2tor
bash scripts/setup.sh
```

Setup membuat `.venv`, menginstal dependensi, dan membuat `.env` dari `.env.example` jika belum ada. `.env` yang sudah ada tidak ditimpa.

### 3. Isi API key

Buka `.env` menggunakan editor teks:

```bash
nano .env
```

Isi baris berikut dengan API key milikmu:

```dotenv
DEEPSEEK_API_KEY=isi_api_key_kamu
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
```

Di nano, tekan **Ctrl+O**, **Enter**, lalu **Ctrl+X** untuk menyimpan dan keluar.

### 4. Jalankan

```bash
bash scripts/run.sh
```

Klik **Start session**. Izinkan akses mikrofon jika diminta. Periksa **System Settings → Privacy & Security → Microphone** untuk izin Terminal yang kamu pakai atau 2tor jika menjalankan paket desktop.

Mac mini membutuhkan mikrofon eksternal, misalnya USB microphone atau headset. Pilih perangkat input/output yang sesuai pada pengaturan suara macOS. Headphone membantu mengurangi echo.

Untuk pemakaian berikutnya, cukup jalankan `bash scripts/run.sh` dari folder proyek. Setelah mengubah `.env`, tutup dan buka ulang 2tor.

## Menjalankan di Ubuntu 26.04 LTS

### 1. Salin source proyek

Salin folder source 2tor ke laptop, misalnya ke `~/projects/2tor`. Jangan menggunakan `.venv` atau hasil build macOS di Ubuntu: dependensi dan binary harus dibuat ulang pada Ubuntu.

### 2. Instal dependensi sistem

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates nano libportaudio2 libglib2.0-0t64 libgl1 libegl1 libxcb-cursor0 libxkbcommon-x11-0
```

`libportaudio2` digunakan untuk input/output audio. Nama paket GLib pada Ubuntu 26.04 adalah [libglib2.0-0t64](https://packages.ubuntu.com/resolute/libglib2.0-0t64).

### 3. Instal uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Buka ulang Terminal, lalu periksa:

```bash
uv --version
```

### 4. Setup dan isi API key

Sesuaikan lokasi folder jika source disimpan di tempat lain:

```bash
cd ~/projects/2tor
bash scripts/setup.sh
nano .env
```

Isi `DEEPSEEK_API_KEY` seperti pada panduan macOS. Untuk X390, gunakan konfigurasi awal berikut:

```dotenv
STT_MODEL=small_streaming
STT_END_PAUSE_SECONDS=1.2
MOONSHINE_ORT_SINGLE_THREAD=1
TTS_MODEL=KittenML/kitten-tts-nano-0.8-int8
TTS_THREADS=2
TTS_SPEED=0.95
PROACTIVE_SECONDS=90
```

Jika sedang menjalankan pekerjaan berat, coba `TTS_THREADS=1` dan `PROACTIVE_SECONDS=300`. Small dipilih untuk menyeimbangkan akurasi dan sumber daya; jika X390 terasa berat, pilih Tiny melalui kontrol Recognition atau `.env`.

### 5. Jalankan dan periksa audio

```bash
bash scripts/run.sh
```

Klik **Start session**, lalu berbicara dalam bahasa Inggris. Periksa input/output di **Settings → Sound** jika mikrofon atau suara belum bekerja.

Uji mesin suara lokal pada X390 tanpa mikrofon atau API:

```bash
bash scripts/run.sh --voice-test
```

Untuk pemakaian berikutnya, cukup jalankan `bash scripts/run.sh` dari folder proyek.

## Pengaturan mikrofon, speaker, dan .env

Lihat daftar perangkat:

```bash
bash scripts/run.sh --devices
```

Isi `.env` dengan indeks atau nama perangkat dari hasil perintah tersebut, atau biarkan kosong untuk default sistem:

```dotenv
INPUT_DEVICE=
OUTPUT_DEVICE=
```

Indeks dapat berubah setelah headset dicabut atau sistem di-restart. Untuk USB microphone, 2tor mengambil audio pada sample rate native perangkat; Moonshine menangani resampling.

Environment variable yang sudah di-export memiliki prioritas di atas `.env`. Aplikasi mencari `.env` di working directory, lalu `~/.config/2tor/.env` jika file pertama tidak ada. `scripts/run.sh` selalu berpindah ke folder proyek sehingga memakai `.env` proyek. Untuk memilih file secara eksplisit:

```bash
bash scripts/run.sh --env /lokasi/file/.env
```

Unduhan model pertama membutuhkan koneksi ke Moonshine/Hugging Face dan bisa memakan beberapa menit. Setelah diunduh, STT/TTS bekerja lokal; DeepSeek tetap memerlukan internet.

Gunakan script setup untuk instalasi KittenTTS. Script memasang `kittenml==0.9.4 --no-deps` dengan dependensi ONNX yang dideklarasikan proyek. Instalasi kittenml biasa menarik PyTorch dan stack KittenTTS 2 yang jauh lebih besar; cara ringan ini mengikuti [panduan upstream](https://github.com/KittenML/KittenTTS/blob/main/docs/onnx-models.md).

## Troubleshooting

| Masalah | Langkah pengecekan |
|---|---|
| `uv: command not found` | Buka ulang Terminal dan periksa PATH melalui panduan resmi uv |
| `.venv/bin/python` tidak ditemukan | Jalankan `bash scripts/setup.sh` pada OS yang sedang dipakai |
| `No module named kittenml` | Jalankan ulang setup; script memasang paket ONNX secara khusus |
| API key kosong / invalid | Periksa `.env` yang dipakai, isi key, lalu restart aplikasi |
| DeepSeek HTTP 402 | Periksa saldo akun API DeepSeek |
| Rate limit / koneksi terputus | Periksa koneksi, tunggu sebentar, lalu kirim ulang pesan |
| Model loading lama | Tunggu unduhan pertama selesai; periksa koneksi ke server model |
| Tidak ada transkrip | Pastikan sesi aktif, Focus tidak aktif, AI selesai menjawab, dan input audio benar |
| Tidak ada suara | Periksa volume/output di OS dan nilai `OUTPUT_DEVICE` |
| Mikrofon macOS ditolak | Periksa izin Microphone untuk Terminal/2tor, lalu restart aplikasi |
| PortAudio tidak tersedia di Ubuntu | Instal `libportaudio2`, lalu jalankan ulang |
| Plugin Qt `xcb` gagal | Pasang dependensi Ubuntu di atas; jalankan dalam sesi desktop grafis |
| Laptop tersendat / backlog audio | Pakai `tiny_streaming`, coba `TTS_THREADS=1`, dan kurangi pekerjaan berat bersamaan |
| Sesi dijeda karena RAM | Tutup dan buka ulang aplikasi; benchmark konfigurasi model sebelum memakainya |
| Paket desktop tidak menemukan key | Isi `~/.config/2tor/.env`, atau gunakan `--env` eksplisit |

## Conversation

- **Recognition**: pilih **Small · more accurate** untuk akurasi lebih baik, atau
  **Tiny · lower CPU** untuk beban CPU lebih rendah. Pause session sebelum mengganti
  model; model sebelumnya dilepas agar tidak ada dua model STT dimuat bersamaan.
  Klik Start session untuk memuat pilihan baru. Saat aplikasi dibuka ulang, pilihan
  mengikuti `STT_MODEL` di `.env`. Instalasi lama dengan `.env` Tiny tidak otomatis
  berubah: pilih Small di UI atau ubah baris `.env` menjadi `STT_MODEL=small_streaming`.
- **Voice**: pilih Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, atau Leo.
  Pilihan langsung dipakai pada jawaban berikutnya; jawaban yang sudah berjalan
  tetap memakai suara sebelumnya. Saat aplikasi dibuka ulang, default kembali
  mengikuti `TTS_VOICE` di `.env`.
- **Listen to sample**: dengarkan contoh bahasa Inggris menggunakan suara yang
  sedang dipilih. Bisa dicoba sebelum Start session, tanpa API key atau mikrofon.
  Contoh dibuat TTS lokal; pertama kali digunakan tetap perlu mengunduh model.
  Mikrofon dijeda selama preview dan dibuka kembali jika sesi sebelumnya aktif.
  Preview tidak masuk riwayat chat. Gunakan Stop reply untuk menghentikan contoh;
  nonaktifkan Focus / mute sebelum mencoba suara.
- Start session: load voices, open microphone, greet you and open an English topic.
- Hands-free listening: partial transcript appears while you talk. A completed
  segment waits another 1.2 seconds by default so short pauses can join into one turn.
  Set `STT_END_PAUSE_SECONDS` between 0.5 and 3 to adjust this extra wait.
- AI response: DeepSeek streams with thinking disabled; complete sentences go to
  a bounded synthesis queue. Audio starts before the whole reply is generated.
- Stop reply: stops playback and cancels the API request; listening resumes after
  the current short native synthesis finishes. Native inference cannot be
  interrupted mid-call. Spoken interruption detection is not implemented yet.
- Capture is gated while AI is generating/speaking, with an additional 350ms echo
  delay before listening resumes. Use Stop reply to take the next turn.
- Quiet check-in: after 90 seconds, AI gently checks in or opens a fresh topic.
  Unanswered prompts back off to 180 and 360 seconds, then stop until you talk.
  Choose a longer interval while working. New topic starts one on demand.
- Focus / mute: stops replies and suppresses microphone processing and proactive
  prompts. Pause closes the microphone. Closing the window exits the app.
- Keep on top: small companion window stays visible. Type a message if useful.
- Gentle corrections are spoken naturally within replies; set CORRECTION_MODE=off
  to disable them. Separate correction cards and pronunciation scoring are not
  implemented in this version.

This app does not read your screen or inspect other applications. Quiet time is
time since detected speech/conversation, not a measurement of your work activity.

## Resource settings

The same low-resource defaults are appropriate for the M4 Pro and X390:

| Setting | Default |
|---|---|
| STT | small_streaming, English, 123M parameters; Tiny option has 34M |
| Extra pause before sending transcript | STT_END_PAUSE_SECONDS=1.2 |
| Native STT inference | MOONSHINE_ORT_SINGLE_THREAD=1 |
| TTS | KittenML/kitten-tts-nano-0.8-int8, 15M parameters |
| TTS CPU threads | 2, sequential ONNX execution, spinning disabled |
| Speech | Jasper, speed 0.95 |
| Raw audio queue | 24 × 100ms chunks maximum |
| Synthesis queue | 3 short text chunks maximum |
| LLM context | Last 20 messages; replies up to 180 output tokens |
| Memory guard | Pause above 2800 MB process-tree RSS, checked every 5 seconds |

STT is reset after completed conversation turns, rather than every VAD segment,
to preserve audio look-behind across short pauses. TTS uses
no CPU memory arena. Quiet STT streams are also reset every 30 seconds. Idle
microphone capture only feeds STT while listening.
For an especially busy X390, try TTS_THREADS=1 or STT_MODEL=tiny_streaming.
Benchmark your selected STT model on the X390. KittenTTS micro/mini are optional
quality alternatives, also ONNX.

Untuk nama/istilah khusus yang sering keliru, isi misalnya
`STT_KEYTERMS=DeepSeek,2tor,Kubernetes`. Maksimal 50 istilah, masing-masing
80 karakter; default kosong. Ini memberi bias pada decoder lokal, bukan
menulis ulang ucapan dengan LLM. Gunakan hanya istilah yang memang kamu pakai.
Jika kalimat terkirim saat kamu masih berpikir, coba `STT_END_PAUSE_SECONDS=1.6`.
Periksa juga pilihan INPUT_DEVICE jika kata pendek seperti “test” tertukar:
pastikan yang dipakai mikrofon/headset fisik yang benar, bukan perangkat audio virtual.

Uji Small + KittenTTS di Mac pada 100 giliran sintetis (enam frasa berulang)
mencatat puncak RSS sekitar 1362 MiB, atau sekitar 1,4 GB. Frasa sintetis
“Test, test.” dikenali sebagai “Test Test”. Median komputasi STT per klip sekitar
447 ms; ini bukan latency percakapan lengkap karena tidak mencakup jeda akhir
ucapan, mikrofon, UI, atau DeepSeek. Hasil ini belum memvalidasi aksenmu maupun
performa X390. Coba frasa yang sama saat status aplikasi **Listening**;
gunakan **Stop reply** jika AI masih menjawab sebelum kamu mengambil giliran.

The memory guard is a soft safety threshold, not a hard OS-level RAM cap: an
allocation can briefly exceed it, and pausing does not unload cached models.
Restart the app to release model memory. We target under 3 GB for the whole app,
but validation must include your microphone, accent and actual Ubuntu version.

## Privacy and history

Microphone audio remains local and is never saved by 2tor. Transcripts, short
conversation history and system prompts go to your configured DeepSeek endpoint.
Keys are read from environment/.env and excluded from Git. API failures display
status hints without echoing the provider's response body or credentials.

History is in memory by default and disappears on exit. SAVE_HISTORY=1 writes
text to `~/.local/share/2tor/history.sqlite3`; no audio is written. Stored history
is an archive, not automatically loaded into new sessions. Clear chat clears the
current context; it does not delete the SQLite archive. Delete that file separately
if you wish to remove saved history.

## Verify and benchmark

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check twotor tests scripts
./scripts/run.sh --smoke-test
./scripts/run.sh --voice-test
.venv/bin/python -m twotor.benchmark --turns 100
# Bandingkan model secara eksplisit tanpa mengubah .env:
.venv/bin/python -m twotor.benchmark --stt-model small_streaming --turns 100 --output artifacts/small.json
.venv/bin/python -m twotor.benchmark --stt-model tiny_streaming --turns 100 --output artifacts/tiny.json
```

The benchmark synthesizes six English phrases including “Test, test.”, feeds them back into local STT
and checks completed transcripts, CPU time, synthesis speed and peak RSS. It
does not open the microphone, play sound or call DeepSeek. JSON reports go in
artifacts/; compare results on both OSes. A full conversational latency figure
also includes speech endpoint detection and your DeepSeek network latency.

## Build dan menjalankan paket desktop

Build harus dijalankan pada OS dan arsitektur tujuan; PyInstaller tidak melakukan cross-compilation. Jalankan perintah berikut dari folder proyek setelah setup.

### macOS: .app

```bash
.venv/bin/python scripts/build.py
```

Hasilnya `dist/2tor.app`. Untuk menjalankan lewat Finder, siapkan konfigurasi desktop:

```bash
mkdir -p ~/.config/2tor
cp .env ~/.config/2tor/.env
chmod 600 ~/.config/2tor/.env
open dist/2tor.app
```

`cp` mengganti konfigurasi desktop dengan `.env` proyek. Setelah itu edit `~/.config/2tor/.env` jika menjalankan lewat Finder. Kamu dapat menyalin `.app` ke Applications. Untuk memilih konfigurasi secara eksplisit lewat Terminal:

```bash
dist/2tor.app/Contents/MacOS/2tor --env "$PWD/.env"
```

Paket menyertakan deskripsi izin mikrofon dan memakai ad-hoc signing lokal; belum dinotarize untuk distribusi publik. Jika macOS memblokir paket yang kamu build sendiri, ikuti opsi **Open Anyway** yang ditawarkan pada **Privacy & Security**.

### Ubuntu: portable atau .deb

Untuk paket portable:

```bash
.venv/bin/python scripts/build.py
dist/2tor/2tor --env "$PWD/.env"
```

Pertahankan seluruh folder `dist/2tor`, jangan hanya menyalin executable-nya.

Untuk membuat dan memasang `.deb`:

```bash
bash scripts/build-deb.sh
sudo apt-get install ./dist/2tor.deb
mkdir -p ~/.config/2tor
cp .env ~/.config/2tor/.env
chmod 600 ~/.config/2tor/.env
```

Buka **2tor** dari menu aplikasi atau jalankan:

```bash
/opt/2tor/2tor
```

Edit `~/.config/2tor/.env` untuk konfigurasi desktop. Bobot model tidak dibundel ke paket; model diunduh saat pertama dipakai dan disimpan dalam cache.

Ubuntu support is implemented but cannot be certified until tested on the X390.
The Linux target is Ubuntu 26.04 LTS on the X390; run setup and the benchmark
there before treating it as verified. The GLib dependency uses the current
[Ubuntu package name](https://packages.ubuntu.com/resolute/libglib2.0-0t64).

## Upstream references

- [Moonshine](https://github.com/moonshine-ai/moonshine),
  [models](https://moonshine-voice.readthedocs.io/en/latest/models/available-models/)
- [KittenTTS](https://github.com/KittenML/KittenTTS),
  [ONNX-only installation](https://github.com/KittenML/KittenTTS/blob/main/docs/onnx-models.md)
- [DeepSeek API](https://api-docs.deepseek.com/api/create-chat-completion/)

The TTS adapter uses the pinned KittenML ONNX inference helpers with explicit
SessionOptions because the upstream constructor does not expose them. Check this
adapter when upgrading kittenml. DeepSeek's current Flash model ID is
deepseek-flash; override DEEPSEEK_MODEL if your provider requires another name.
