#!/usr/bin/env bash
set -euo pipefail
echo "Starting StudyGlance dev (backend + overlay + web)"
(cd backend && uvicorn app.main:app --reload) &
(cd apps/overlay/pyqt && python main.py) &
(cd web/site && npm install && npm run dev) &
wait

