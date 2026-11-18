# Windows setup for the PBH project (PowerShell).  Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
# Idempotent: safe to run again.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# 1. Python >= 3.10 (install with winget if missing)
function Get-Py {
  foreach ($c in @("py -3", "python")) {
    try { $v = & cmd /c "$c -c ""import sys;print(sys.version_info[:2]>=(3,10))"" 2>nul"; if ($v -eq "True") { return $c } } catch {}
  }
  return $null
}
$py = Get-Py
if (-not $py) {
  Write-Host "Python >= 3.10 not found; installing Python 3.12 with winget ..."
  winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
  $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
  $py = Get-Py
  if (-not $py) { throw "Python installation failed; install Python 3.12 manually and re-run." }
}
Write-Host "using: $py"

# 2. virtual environment + packages
if (-not (Test-Path ".venv")) { & cmd /c "$py -m venv .venv" }
$vpy = Join-Path $root ".venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip
& $vpy -m pip install -r requirements.txt

# 3. external data: NANOGrav 15-yr KDE (shipped zip) -> unzip; PBHbounds/ceffyl snapshots are shipped
$ng = Join-Path $root "external\NG15"
if (-not (Test-Path (Join-Path $ng "30f_fs{hd}_ceffyl\density.npy"))) {
  Expand-Archive -Path (Join-Path $ng "NANOGrav15yr_KDE-FreeSpectra_v1.0.0.zip") -DestinationPath $ng -Force
}

}

# 5. environment check
$env:PYTHONUTF8 = "1"
Push-Location pbh
& $vpy regenerate.py --check
Pop-Location
Write-Host "`nSetup finished. Next:  .venv\Scripts\python.exe pbh\regenerate.py  (run inside pbh\)  -- see README.md"
