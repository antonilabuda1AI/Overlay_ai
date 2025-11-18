#!/usr/bin/env bash
set -euo pipefail
echo "Building StudyGlance (PyInstaller)"
pip install -r backend/requirements.txt
pip install -r apps/overlay/requirements.txt
pyinstaller --onefile --name studyglance-backend backend/app/main.py
pyinstaller --onefile --name studyglance apps/overlay/pyqt/main.py

