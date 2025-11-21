
OverlayAI â€” Usage Guide
=========================

Prerequisites
- Windows/macOS/Linux
- Tesseract OCR installed and available in PATH (or specify path in Settings)
- OpenAI API key set in environment: `OPENAI_API_KEY`

Install
- Windows: download OverlayAI-Setup.exe and run installer (placeholder link on Download page)
- macOS: drag OverlayAI.app to Applications; grant screen capture permission
- Linux: run OverlayAI-AppImage; grant screen capture permission if needed

First Run
1) Launch backend (auto-started by overlay on first run) or run manually: `OverlayAI-backend`
   - Dev: `cd backend && uvicorn app.main:app --reload`
2) Launch overlay: `OverlayAI`
   - Dev: `cd apps/overlay/pyqt && python main.py`
3) Set OCR languages in Settings (default `eng+pol`)
4) Press Ctrl+L to start Live. App observes the screen in real time (no recording).
5) Press Ctrl+M to mark moments; type notes.
6) Ask questions in the input (e.g., â€śsummarize the last slideâ€ť, â€śwhat was the API rate limit shown?â€ť).
7) Export session: Settings â†’ Export â†’ JSON/Markdown.

Privacy
- Frames are never stored; only OCR text with timestamps lives in your local session file.
- Only text is sent to OpenAI when you ask questions.

Troubleshooting
- Tesseract not found â†’ specify path in Settings or ensure itâ€™s on PATH.
- High CPU â†’ lower OCR interval, reduce max ROIs, or enable Eco mode.
- No text captured â†’ check screen permission (macOS), or choose correct monitor/bbox in Live settings.

HOW TO RUN (DEV)
- `cd backend && uvicorn app.main:app --reload`
- `cd apps/overlay/pyqt && python main.py`
- `cd web/site && npm install && npm run dev`

