
StudyGlance — Usage Guide
=========================

Prerequisites
- Windows/macOS/Linux
- Tesseract OCR installed and available in PATH (or specify path in Settings)
- OpenAI API key set in environment: `OPENAI_API_KEY`

Install
- Windows: download StudyGlance-Setup.exe and run installer (placeholder link on Download page)
- macOS: drag StudyGlance.app to Applications; grant screen capture permission
- Linux: run StudyGlance-AppImage; grant screen capture permission if needed

First Run
1) Launch backend (auto-started by overlay on first run) or run manually: `studyglance-backend`
   - Dev: `cd backend && uvicorn app.main:app --reload`
2) Launch overlay: `studyglance`
   - Dev: `cd apps/overlay/pyqt && python main.py`
3) Set OCR languages in Settings (default `eng+pol`)
4) Press Ctrl+L to start Live. App observes the screen in real time (no recording).
5) Press Ctrl+M to mark moments; type notes.
6) Ask questions in the input (e.g., “summarize the last slide”, “what was the API rate limit shown?”).
7) Export session: Settings → Export → JSON/Markdown.

Privacy
- Frames are never stored; only OCR text with timestamps lives in your local session file.
- Only text is sent to OpenAI when you ask questions.

Troubleshooting
- Tesseract not found → specify path in Settings or ensure it’s on PATH.
- High CPU → lower OCR interval, reduce max ROIs, or enable Eco mode.
- No text captured → check screen permission (macOS), or choose correct monitor/bbox in Live settings.

HOW TO RUN (DEV)
- `cd backend && uvicorn app.main:app --reload`
- `cd apps/overlay/pyqt && python main.py`
- `cd web/site && npm install && npm run dev`

