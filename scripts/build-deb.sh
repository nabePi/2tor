#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$(uname -s)" != Linux ]; then echo 'Build this package on Ubuntu.'; exit 1; fi
.venv/bin/python scripts/build.py
stage=build/deb
mkdir -p "$stage/DEBIAN" "$stage/opt/2tor" "$stage/usr/share/applications"
cp -a dist/2tor/. "$stage/opt/2tor/"
cat > "$stage/DEBIAN/control" <<EOF
Package: 2tor
Version: 0.1.0
Architecture: $(dpkg --print-architecture)
Maintainer: 2tor
Depends: libportaudio2, libglib2.0-0t64 | libglib2.0-0, libgl1, libegl1, libxcb-cursor0, libxkbcommon-x11-0
Description: Local speech English conversation companion using DeepSeek
EOF
cat > "$stage/usr/share/applications/2tor.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=2tor
Comment=English while you work
Exec=/opt/2tor/2tor
Terminal=false
Categories=Education;Languages;
EOF
dpkg-deb --build --root-owner-group "$stage" dist/2tor.deb
