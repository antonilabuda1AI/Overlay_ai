param()
Write-Host "Starting StudyGlance dev (backend + overlay + web)"
Start-Process powershell -ArgumentList "-NoExit","-Command","cd backend; uvicorn app.main:app --reload" | Out-Null
Start-Process powershell -ArgumentList "-NoExit","-Command","cd apps/overlay/pyqt; python main.py" | Out-Null
Start-Process powershell -ArgumentList "-NoExit","-Command","cd web/site; npm install; npm run dev" | Out-Null

