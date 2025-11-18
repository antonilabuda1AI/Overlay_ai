param()
Write-Host "Building StudyGlance overlay (Windows)"

# Pick Python: prefer repo venv, then system python
$venvPy = Join-Path $PSScriptRoot "..\backend.venv\Scripts\python.exe" | Resolve-Path -ErrorAction SilentlyContinue
if ($venvPy) { $py = $venvPy.Path } else { $py = "python" }
Write-Host "Using Python:" $py

Write-Host "Installing PyInstaller..."
& $py -m pip install --upgrade pip | Out-Null
& $py -m pip install pyinstaller | Out-Null

# Ensure deps installed
Write-Host "Installing overlay dependencies..."
& $py -m pip install -r (Join-Path $PSScriptRoot "..\apps\overlay\requirements.txt")

Write-Host "Running PyInstaller..."
& $py -m PyInstaller --noconsole --onefile --name StudyGlance --collect-all PyQt6 overlay_qna_prod.py

if (Test-Path "dist\StudyGlance.exe") {
  Write-Host "Done. Binary at dist\StudyGlance.exe"
} else {
  Write-Warning "Build did not produce dist\StudyGlance.exe. Check PyInstaller output above."
}
