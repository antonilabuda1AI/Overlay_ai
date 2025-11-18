StudyGlance
===========

Privacy-first, study-oriented overlay assistant. It observes your screen in real time (no recording), performs OCR on meaningful changes, stores only text locally per session, and lets you ask questions with cited timeline snippets.

Key features
- Cross-platform: Windows, macOS, Linux
- Overlay bar + answer bubble UI (PyQt)
- Hotkeys: Ctrl+L (Live), Ctrl+M (Mark), Ctrl+H (History), Ctrl+Q (Quit)
- Real-time OCR with delta gating (mss + Tesseract)
- Local session SQLite storage; one file per meeting
- RAG over OCR timeline; OpenAI-only for embeddings + chat (text only)
- Minimal website (Vite + React + Tailwind)
- Packaging scripts for Win/macOS/Linux

Quickstart
- Prereqs: Tesseract installed, OpenAI API key set
- Backend (dev): `cd backend && uvicorn app.main:app --reload`
- Overlay (dev): `cd apps/overlay/pyqt && python main.py`
- Website (dev): `cd web/site && npm install && npm run dev`

Offline/No-quota mode
- Backend: set `LLM_FORCE_LOCAL=1` to force local embeddings/chat stubs (no OpenAI calls).
- Standalone overlay script (`overlay_qna_prod.py`): set `LLM_OFFLINE=1` to run fully offline with local fallbacks.

See USAGE.md for end-user instructions.

Privacy
- Frames are never stored or uploaded. Only extracted text is saved locally. Only text is sent to OpenAI when you ask questions.

Development
- Python 3.10+
- Install backend requirements: `pip install -r backend/requirements.txt`
- Run tests: `cd backend && pytest`

Packaging
- Scripts under `scripts/` for PyInstaller and platform packaging. These are provided with sensible defaults and placeholders you can adapt.
