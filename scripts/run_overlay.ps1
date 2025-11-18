param()
Write-Host "Starting StudyGlance overlay"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
  Write-Host "No .venv in current folder. Using system python."
  $py = "python"
} else {
  $py = ".venv\Scripts\python.exe"
}
& $py overlay_qna_prod.py

