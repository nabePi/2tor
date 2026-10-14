"""Build on the destination OS. Apple Silicon on macOS; x86_64 on Ubuntu."""

import os
import sys
from pathlib import Path

from PyInstaller.__main__ import run

root = Path(__file__).resolve().parent.parent
os.chdir(root)
args = [
    str(root / "scripts/launcher.py"),
    "--paths=" + str(root),
    "--name=2tor",
    "--onedir",
    "--noconfirm",
    "--clean",
    "--windowed",
    "--collect-all=moonshine_voice",
    "--collect-all=faster_whisper",
    "--collect-all=ctranslate2",
    "--collect-all=kokoro_onnx",
    "--collect-all=espeakng_loader",
    "--collect-data=kittenml",
    "--collect-data=phonemizer",
    "--copy-metadata=kittenml",
    "--hidden-import=kittenml.kittentts_legacy.onnx_model",
    "--hidden-import=phonemizer.backend.espeak",
    "--hidden-import=onnxruntime",
    "--exclude-module=torch",
    "--exclude-module=torchaudio",
    "--exclude-module=transformers",
    "--exclude-module=pytest",
]
if sys.platform == "darwin":
    args += ["--osx-bundle-identifier=com.twotor.desktop"]
run(args)
if sys.platform == "darwin":
    import plistlib
    import subprocess

    plist = root / "dist/2tor.app/Contents/Info.plist"
    info = plistlib.loads(plist.read_bytes())
    info["NSMicrophoneUsageDescription"] = (
        "2tor transcribes your voice locally for English conversation."
    )
    info["NSHighResolutionCapable"] = True
    plist.write_bytes(plistlib.dumps(info))
    subprocess.run(
        ["codesign", "--force", "--deep", "--sign", "-", str(root / "dist/2tor.app")], check=True
    )
    print(f"Built {root / 'dist/2tor.app'} (local ad-hoc signing; not notarized)")
else:
    print(f"Built {root / 'dist/2tor/2tor'}")
