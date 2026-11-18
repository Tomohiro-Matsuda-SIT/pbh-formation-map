#!/usr/bin/env bash
# POSIX setup (Git Bash on Windows, WSL, Linux, macOS).  Run from the repository root:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=""
for c in python3 python "py -3"; do
  if $c -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)' 2>/dev/null; then PY="$c"; break; fi
done
[ -z "$PY" ] && { echo "Python >= 3.10 not found. On Windows run setup.ps1 (installs via winget)."; exit 1; }
[ -d .venv ] || $PY -m venv .venv
if [ -x .venv/Scripts/python.exe ]; then VPY=.venv/Scripts/python.exe; else VPY=.venv/bin/python; fi
$VPY -m pip install --upgrade pip
$VPY -m pip install -r requirements.txt
NG=external/NG15
[ -f "$NG/30f_fs{hd}_ceffyl/density.npy" ] || (cd $NG && $VPY -c "import zipfile; zipfile.ZipFile('NANOGrav15yr_KDE-FreeSpectra_v1.0.0.zip').extractall('.')")
(cd pbh && PYTHONUTF8=1 ../$VPY regenerate.py --check)
echo "Setup finished. Next: (cd pbh && ../$VPY regenerate.py)  -- see README.md"
