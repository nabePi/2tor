# 2tor

Aplikasi desktop untuk menemani kamu bekerja sambil melatih percakapan bahasa Inggris. Kamu berbicara lewat mikrofon, AI menjawab dengan suara, dan AI bisa membuka topik saat kamu diam.

**Mikrofon → STT lokal → DeepSeek API → TTS lokal atau Edge TTS online → speaker/headphone.**

| Komponen | Default |
|---|---|
| Desktop | Python 3.11 + PySide6 Qt Widgets |
| STT | faster-whisper `small.en`, CPU INT8, beam search 5 |
| Deteksi ucapan | VAD native Moonshine, tanpa memuat model ASR Moonshine pada mode Whisper |
| LLM | DeepSeek `deepseek-flash`, streaming, thinking dinonaktifkan |
| TTS | Kokoro v1.0 INT8 melalui ONNX Runtime, suara Heart, tempo 1.05× |

GPU dan PyTorch tidak diperlukan. Model suara diunduh saat pertama dipakai, kemudian digunakan dari cache lokal. DeepSeek tetap membutuhkan internet dan API key.

**Edge TTS** tersedia sebagai pilihan tambahan. Engine ini memakai layanan online
Microsoft, tanpa API key Microsoft dan tanpa perlu menginstal browser Edge.
Teks yang akan diucapkan dikirim ke Microsoft; mikrofon/STT tetap lokal.
Kitten dan Kokoro tetap tersedia untuk TTS lokal.

Target perangkat: **Mac mini M4 Pro, RAM 24 GB**, serta **Lenovo ThinkPad X390, Intel Core i5 generasi 8 vPro, RAM 16 GB, Ubuntu 26.04 LTS**. UI, model lokal, dan `.app` diuji di macOS; pengujian langsung di X390 masih diperlukan.

## Menjalankan di macOS

### 1. Instal uv

Jika Homebrew tersedia:

```bash
brew install uv
```

Alternatif tanpa Homebrew:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Buka ulang Terminal, lalu periksa `uv --version`. Lihat [panduan resmi uv](https://docs.astral.sh/uv/getting-started/installation/) jika PATH belum tersedia. Setup menyiapkan Python 3.11 melalui uv.

### 2. Setup dan API key

```bash
cd /Users/wahyusaid/projects/2tor
bash scripts/setup.sh
nano .env
```

Isi `DEEPSEEK_API_KEY` dengan key milikmu. Di nano: **Ctrl+O → Enter → Ctrl+X**. Setup membuat `.venv` dan `.env` jika belum ada, serta mempertahankan `.env` yang sudah ada.

### 3. Jalankan

```bash
bash scripts/run.sh
```

Klik **Start session**. Izinkan akses mikrofon saat diminta. Periksa **System Settings → Privacy & Security → Microphone** untuk Terminal atau 2tor. Mac mini membutuhkan mikrofon eksternal/headset; pilih perangkat yang benar pada pengaturan suara. Headphone membantu mengurangi echo.

Untuk penggunaan selanjutnya, cukup jalankan script run dari folder proyek. Restart aplikasi setelah mengubah `.env`.

## Menjalankan di Ubuntu 26.04 LTS

Salin source proyek, misalnya ke `~/projects/2tor`. Jangan menyalin `.venv` atau menggunakan binary macOS pada Ubuntu; buat ulang dependensi pada OS tujuan.

### 1. Dependensi sistem dan uv

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates nano libportaudio2 libglib2.0-0t64 libgl1 libegl1 libxcb-cursor0 libxkbcommon-x11-0
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Buka ulang Terminal dan periksa `uv --version`. Paket GLib untuk Ubuntu 26.04 memakai nama [libglib2.0-0t64](https://packages.ubuntu.com/resolute/libglib2.0-0t64).

### 2. Setup dan jalankan

```bash
cd ~/projects/2tor
bash scripts/setup.sh
nano .env
# Isi DEEPSEEK_API_KEY, simpan, lalu:
bash scripts/run.sh
```

Periksa input/output pada **Settings → Sound**. Pada X390 yang sedang menjalankan pekerjaan berat, coba **Whisper Base · faster** dan `STT_THREADS=1` atau `TTS_THREADS=1`. Base mengutamakan kecepatan; uji kembali akurasinya pada ucapanmu.

## Memperbarui instalasi lama

```bash
git pull origin main
bash scripts/setup.sh
```

Setup tidak menimpa `.env` lama. Untuk mencoba mesin baru tanpa mengedit konfigurasi, **Pause session**, pilih **Recognition: Whisper Small · accuracy**, lalu pilih **Kokoro** pada baris Voice dan suara **Heart**. Pilihan engine dapat diubah ketika sesi berhenti.

Untuk menjadikan konfigurasi baru sebagai default setelah restart, sesuaikan baris berikut di `.env` yang dipakai:

```dotenv
STT_MODEL=whisper_small_en
STT_THREADS=2
STT_END_PAUSE_SECONDS=0.65
TTS_MODEL=kokoro_int8
TTS_VOICE=Heart
TTS_SPEED=1.05
TTS_THREADS=2
```

KittenTTS Nano dan Moonshine tetap tersedia sebagai pilihan ringan. Suara Kitten dan Kokoro berbeda; nama suara harus sesuai engine. Jangan menggunakan `TTS_VOICE=Jasper` dengan Kokoro.

### Memakai Edge TTS

Setelah `git pull origin main` dan `bash scripts/setup.sh`, restart aplikasi.
Jika memakai `.app` atau `.deb`, rebuild paket setelah memperbarui source.

1. Klik **Pause session**.
2. Pada baris Voice, pilih **Edge (online)**.
3. Pilih suara, misalnya **Jenny**, **Emma**, atau **Guy**, lalu klik **Listen to sample**.
4. Sesuaikan **Speaking speed**; default saat beralih ke Edge adalah 1.0×.
5. Klik **Start session** untuk ngobrol menggunakan suara tersebut.

Untuk menjadikan Edge default setelah restart:

```dotenv
TTS_MODEL=edge_tts
TTS_VOICE=Jenny
TTS_SPEED=1.0
```

Edge tidak mengunduh bobot model TTS. Preview bisa dicoba tanpa API key DeepSeek,
tetapi tetap memerlukan internet dan mengirim teks contoh ke Microsoft.
**Stop reply** membatalkan koneksi sintesis yang sedang berjalan.
Jika layanan tidak tersedia, aplikasi menampilkan error; pilih Kokoro/Kitten untuk
kembali memakai TTS lokal. Tidak ada penggantian engine otomatis tanpa pilihanmu.

## Tes mikrofon sebelum ngobrol

1. Pause sesi, lalu pilih mikrofon fisik pada dropdown **Microphone**. Hindari perangkat audio virtual jika ingin memakai headset/mikrofon langsung.
2. Klik **Test mic**. Mode ini hanya memuat STT, tanpa API key, TTS, penyimpanan audio, atau request DeepSeek.
3. Ucapkan **“test test test”**, kemudian **“best best best”**, dan lihat transkripnya.
4. Perhatikan meter: **Low / quiet** berarti level rendah/hening, **Clipping** berarti sinyal mendekati batas maksimum. Perbaiki posisi mikrofon atau input volume pada OS bila diperlukan. Meter ini bukan pengukuran kualitas aksen atau noise.
5. Klik **Stop test**, lalu **Start session** untuk percakapan.

Transkrip mengikuti hasil pengenalan suara. Aplikasi tidak mengganti “best” menjadi “test” atau menulis ulang ucapan menggunakan LLM untuk menutupi kesalahan.

Daftar perangkat juga bisa dilihat dari Terminal:

```bash
bash scripts/run.sh --devices
```

`INPUT_DEVICE` dan `OUTPUT_DEVICE` menerima indeks atau nama perangkat; kosong berarti default OS. Indeks dapat berubah setelah headset dicabut/restart. Audio diambil pada sample rate native perangkat dan di-resample oleh pipeline lokal.

## Kontrol percakapan dan suara

| Kontrol | Fungsi |
|---|---|
| Start / Pause session | Memulai percakapan atau menutup mikrofon |
| Stop reply | Membatalkan request/jawaban dan menghentikan suara |
| Focus / mute | Menghentikan jawaban, pemrosesan mikrofon, dan check-in |
| Voice: Edge / Kokoro / Kitten | Memilih engine TTS ketika sesi berhenti |
| Dropdown suara | Mengganti suara pada jawaban berikutnya |
| Listen to sample | Mendengarkan contoh tanpa key DeepSeek/mikrofon; Edge memakai internet |
| Speaking speed | Mengatur tempo 0.7×–1.3×, berlaku pada jawaban/preview berikutnya |
| Recognition | Whisper Small/Base atau Moonshine Small/Tiny; ubah saat sesi berhenti |
| Microphone / Test mic | Memilih input dan menguji transkrip lokal |
| Quiet check-in / New topic | Mengatur interval ajakan ngobrol atau membuka topik sekarang |
| Keep on top | Menjaga jendela terlihat di atas aplikasi lain |
| Send / Clear chat | Mengirim teks atau membersihkan konteks sesi aktif |

Suara Kokoro: **Heart, Bella, Nicole, Sarah, Adam, Michael, Fenrir, Emma**. Emma memakai pelafalan Inggris British; pilihan lainnya memakai English US. Kitten: **Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo**.

Suara Edge: **Jenny, Emma, Aria, Guy, Andrew, Brian, Ava, Christopher** (English US),
**Sonia, Ryan** (English British), serta **Natasha, William** (English Australia).
Nama Emma di Edge dan Kokoro merujuk ke suara yang berbeda. Pilihan layanan
dapat berubah; jika suatu suara tidak tersedia, coba suara Edge lainnya.

Whisper mendekode segmen setelah jeda ucapan, sehingga tidak menampilkan transkrip kata demi kata saat kamu masih berbicara. Model `small.en` memakai CPU INT8 dan beam search; latensinya dapat lebih tinggi daripada Moonshine. Base tersedia untuk CPU laptop yang sibuk. Istilah khusus dapat dibantu melalui `STT_KEYTERMS`, default kosong.

Stream dipertahankan selama satu giliran, dengan tambahan jeda default 0.65 detik setelah segmen selesai agar jeda pendek dapat digabung. Naikkan `STT_END_PAUSE_SECONDS` jika kalimat terkirim terlalu cepat. Jeda ini menambah waktu sebelum respons.

Jawaban DeepSeek dikirim ke TTS per kalimat. Kalimat berikutnya disintesis **sambil kalimat sebelumnya diputar**, dengan antrean audio terbatas, agar jeda antarkalimat tidak menunggu sintesis baru. AI diminta membuka dengan kalimat singkat. Waktu respons tetap mencakup deteksi akhir ucapan, STT, jaringan DeepSeek, dan sintesis pertama.

Mikrofon diabaikan selama AI menjawab untuk menghindari echo. Mulai berbicara saat status **Listening**; gunakan **Stop reply** jika ingin menyela. Interupsi otomatis lewat suara belum tersedia. Inference native perlu menyelesaikan potongan yang sedang berjalan sebelum pembatalan selesai; setelah respons normal ada jeda echo 350 ms.

Check-in default 90 detik. Ajakan yang tidak dijawab memperpanjang jeda menjadi 180 lalu 360 detik, kemudian berhenti sampai kamu berbicara kembali. Focus menonaktifkannya. AI tidak membaca layar atau aplikasi lain. Koreksi grammar disampaikan ringan dalam jawaban; penilaian pelafalan khusus belum tersedia.

## Konfigurasi .env

| Variabel | Default / pilihan |
|---|---|
| `DEEPSEEK_API_KEY` | Wajib untuk percakapan; tes mic/preview tidak memerlukan key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com`, harus HTTPS |
| `DEEPSEEK_MODEL` | `deepseek-flash` |
| `STT_MODEL` | `whisper_small_en`; alternatif `whisper_base_en`, `small_streaming`, `tiny_streaming` |
| `STT_THREADS` | `2`, rentang 1–4 untuk Whisper |
| `STT_END_PAUSE_SECONDS` | `0.65`, rentang 0.5–3 |
| `STT_KEYTERMS` | Istilah dipisahkan koma, maksimal 50 istilah × 80 karakter; kosong untuk percakapan umum |
| `MOONSHINE_ORT_SINGLE_THREAD` | `1`, inference native Moonshine satu thread |
| `TTS_MODEL` | `kokoro_int8`; alternatif `edge_tts` atau Kitten Nano/Micro/Mini ONNX |
| `TTS_VOICE` | `Heart`, harus cocok dengan engine |
| `TTS_SPEED` | `1.05`, rentang 0.7–1.3 |
| `TTS_THREADS` | `2`, rentang 1–4 |
| `INPUT_DEVICE`, `OUTPUT_DEVICE` | Kosong untuk default OS, atau nama/indeks perangkat |
| `PROACTIVE_SECONDS` | `90`, rentang 30–3600 |
| `CORRECTION_MODE` | `gentle` atau `off` |
| `SAVE_HISTORY` | `0`; `1` untuk arsip teks lokal |
| `MEMORY_LIMIT_MB` | `2800`, maksimal konfigurasi 3000 |

Environment yang sudah di-export memiliki prioritas di atas `.env`. Aplikasi mencari `.env` di working directory, lalu `~/.config/2tor/.env`. Script run berpindah ke folder proyek sehingga memakai `.env` proyek. File eksplisit:

```bash
bash scripts/run.sh --env /lokasi/file/.env
```

Pilihan UI berlaku pada sesi berjalan. Default saat restart mengikuti `.env`.

## RAM, CPU, dan verifikasi

Hanya satu model ASR dipakai: pada mode Whisper, Moonshine hanya mendeteksi ucapan tanpa bobot model ASR-nya. STT direset saat giliran selesai serta saat diam, dan segmen Whisper dibatasi 20 detik. Antrean raw audio maksimal 24 × 100 ms, antrean teks tiga potongan, dan antrean audio hasil sintesis dua potongan. TTS memakai ONNX sequential, dua thread, spinning dinonaktifkan. Konteks LLM maksimal 20 pesan terakhir, output maksimal 180 token.

Uji Mac pada **48 giliran sintetis** dengan Whisper Small INT8 + Kokoro INT8 mencatat puncak RSS sekitar **1278 MiB / 1.34 GB**. Sintesis lebih cepat daripada durasi audio. Uji awal mencatat sekitar 0.2–0.3 detik untuk sintesis frasa pendek, sedangkan komputasi STT sekitar 1.3–1.4 detik per klip. Ini bukan hasil pengujian aksenmu, kualitas natural suara, latency jaringan, mikrofon, UI, maupun performa X390.

Target seluruh aplikasi kurang dari 3 GB. Pemantau RSS process tree setiap lima detik menjeda sesi di atas batas. Ini batas lunak, bukan pembatas OS; alokasi dapat sesaat melampaui ambang dan pause tidak melepas semua cache. Tutup aplikasi untuk melepas memori.

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check twotor tests scripts
bash scripts/run.sh --smoke-test
bash scripts/run.sh --voice-test
.venv/bin/python -m twotor.benchmark --turns 48
```

Smoke test membuka UI sebentar tanpa audio/API. Voice test menyintesis dan mentranskripsikan tiga frasa tanpa mikrofon, playback, atau DeepSeek. Benchmark lengkap memakai delapan frasa termasuk pasangan “test” dan “best”; laporan JSON berada di `artifacts/benchmark.json`.

Pengecualian jaringan: jika `.env` memilih Edge, voice test/benchmark memakai
Microsoft untuk menyintesis frasa uji, tanpa memanggil DeepSeek. Dengan Kokoro atau
Kitten, sintesis tetap lokal setelah model diunduh.

Perbandingan model:

```bash
.venv/bin/python -m twotor.benchmark --stt-model whisper_base_en --output artifacts/base.json
.venv/bin/python -m twotor.benchmark --stt-model whisper_small_en --output artifacts/small.json
```

## Build paket desktop

Build pada OS/arsitektur tujuan; PyInstaller tidak cross-compile. Setelah setup:

```bash
.venv/bin/python scripts/build.py
```

### macOS .app

Hasil: `dist/2tor.app`. Siapkan konfigurasi untuk Finder:

```bash
mkdir -p ~/.config/2tor
cp .env ~/.config/2tor/.env
chmod 600 ~/.config/2tor/.env
open dist/2tor.app
```

`cp` mengganti konfigurasi desktop. Edit file tersebut untuk pemakaian Finder. Paket boleh disalin ke Applications. Untuk file konfigurasi eksplisit:

```bash
dist/2tor.app/Contents/MacOS/2tor --env "$PWD/.env"
```

Paket menyertakan deskripsi izin mikrofon dan ad-hoc signing lokal; belum dinotarize untuk distribusi publik. Jika paket yang kamu build sendiri diblokir macOS, ikuti opsi Open Anyway pada Privacy & Security.

### Ubuntu portable / .deb

Portable: pertahankan seluruh folder `dist/2tor`, bukan hanya executable.

```bash
dist/2tor/2tor --env "$PWD/.env"
```

Paket `.deb`:

```bash
bash scripts/build-deb.sh
sudo apt-get install ./dist/2tor.deb
mkdir -p ~/.config/2tor
cp .env ~/.config/2tor/.env
chmod 600 ~/.config/2tor/.env
```

Buka 2tor dari menu aplikasi atau `/opt/2tor/2tor`. Edit `~/.config/2tor/.env` untuk konfigurasi desktop. Model tidak dibundel; Whisper di-cache oleh Hugging Face, Kokoro di `~/.cache/2tor/kokoro-v1.0`, dan model Moonshine di cache library-nya.

## Troubleshooting

| Masalah | Pengecekan |
|---|---|
| uv tidak ditemukan | Buka ulang Terminal; periksa PATH di panduan uv |
| Python/dependensi tidak ditemukan | Jalankan setup ulang pada OS tujuan |
| Suara/model baru tidak muncul | Pull kode, setup ulang, dan rebuild `.app`/`.deb` bila memakai paket |
| Kata “test” menjadi “best” | Pilih Whisper Small, tes mic fisik, periksa level; jangan menganggap uji sintetis menjamin aksen manusia |
| Suara terasa pelan | Coba Kokoro Heart dan Speaking speed 1.05–1.15×; preview terlebih dahulu |
| Edge tidak menghasilkan suara | Periksa internet, coba suara Edge lain, atau kembali ke TTS lokal |
| Tidak ada transkrip | Tunggu Listening, nonaktifkan Focus, pastikan AI selesai dan input benar |
| Low / quiet atau Clipping | Periksa input volume, jarak mikrofon, serta apakah perangkat virtual terpilih |
| API invalid / HTTP 402 | Periksa key/file `.env` yang dipakai atau saldo akun API |
| Rate limit / koneksi putus | Periksa internet, tunggu sebentar, kirim ulang pesan |
| Unduhan awal lambat | Tunggu; akses Hugging Face/Moonshine/GitHub diperlukan untuk model terkait |
| PortAudio / Qt xcb gagal di Ubuntu | Pasang dependensi sistem dan jalankan di sesi desktop grafis |
| Laptop berat / audio backlog | Coba Whisper Base, kurangi thread, atau gunakan opsi Moonshine |
| Batas RAM tercapai | Tutup aplikasi, uji model lebih kecil dengan benchmark |
| Paket desktop tidak menemukan key | Isi `~/.config/2tor/.env` atau pakai `--env` |

Setup masih memasang KittenML dengan `--no-deps` agar pilihan legacy ONNX tidak menarik PyTorch/KittenTTS 2. Jangan memasang kittenml biasa untuk konfigurasi ringan ini.

## Privasi

Audio mikrofon tetap lokal dan tidak disimpan. Mode Test mic dan preview tidak memanggil DeepSeek. Percakapan mengirim transkrip dan konteks pendek ke endpoint yang dikonfigurasi. API key berada di environment/.env dan dikecualikan dari Git.

Saat Edge dipilih, teks jawaban AI atau contoh suara juga dikirim ke layanan
Microsoft untuk sintesis. Audio hasil Edge diterima sebagai MP3 dan di-decode
di memori menjadi PCM mono 24 kHz; aplikasi tidak menyimpan MP3 tersebut.

Riwayat default hanya di memori. `SAVE_HISTORY=1` menyimpan teks di `~/.local/share/2tor/history.sqlite3`; tidak dimuat otomatis pada sesi baru. Clear chat hanya membersihkan konteks aktif. Hapus file SQLite setelah menutup aplikasi jika ingin menghapus arsip.

## Referensi

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx)
- [Edge TTS](https://github.com/rany2/edge-tts)
- [Moonshine](https://github.com/moonshine-ai/moonshine)
- [KittenTTS ONNX ringan](https://github.com/KittenML/KittenTTS/blob/main/docs/onnx-models.md)
- [DeepSeek API](https://api-docs.deepseek.com/api/create-chat-completion/)

Helper KittenML dipin versinya karena adapter mengatur SessionOptions secara eksplisit. Periksa adapter saat memperbarui library. ID model DeepSeek dapat disesuaikan melalui environment.
