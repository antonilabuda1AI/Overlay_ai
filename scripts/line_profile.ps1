param(
  [Parameter(Mandatory=$true)][string]$File,
  [string[]]$Args
)
if (-not (Get-Command kernprof -ErrorAction SilentlyContinue)) {
  Write-Host "kernprof (line_profiler) not found. pip install line_profiler" -ForegroundColor Yellow
  exit 1
}
kernprof -l $File @Args
python -m line_profiler $(Split-Path $File -Leaf).lprof

