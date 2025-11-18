#!/usr/bin/env bash
set -euo pipefail
echo "Building StudyGlance overlay (Linux)"
python -m pip install --upgrade pip
python -m pip install pyinstaller
python -m pip install -r apps/overlay/requirements.txt
pyinstaller --noconsole --onefile --name StudyGlance overlay_qna_prod.py
echo "Done. Binary at dist/StudyGlance"

