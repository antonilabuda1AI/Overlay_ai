#!/usr/bin/env bash
set -euo pipefail
echo "Building StudyGlance overlay (macOS)"
python -m pip install --upgrade pip
python -m pip install pyinstaller
python -m pip install -r apps/overlay/requirements.txt
pyinstaller --windowed --onefile --name StudyGlance overlay_qna_prod.py
echo "Done. App bundle in dist/StudyGlance.app (or standalone if codesigning blocked)."

