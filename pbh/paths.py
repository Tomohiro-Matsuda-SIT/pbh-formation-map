"""
Central place for all file-system locations (handoff 2026-11-15).
Defaults are relative to this file, so the bundle works wherever it is unpacked:
    <bundle>/pbh/                 code + data (this directory, PBH_ROOT)
    <bundle>/external/PBHbounds   PBHbounds snapshot (commit 96bc0911292ca637a6c6a61029afe98a311595ad)
    <bundle>/external/ceffyl      ceffyl data snapshot (commit 0cd2b3ad6f1100d6b3b287bd28718663672262af)
    <bundle>/external/NG15        NANOGrav 15-yr KDE free spectra (Zenodo 10.5281/zenodo.10344086 v1.0.0), unzipped
Each can be overridden with an environment variable of the same name.
NPROC: worker processes for the parallel scans (default: os.cpu_count() - 2, at least 1).
"""
import os
_here = os.path.dirname(os.path.abspath(__file__))
_ext = os.path.normpath(os.path.join(_here, "..", "external"))
PBH_ROOT = os.environ.get("PBH_ROOT", _here)
PBHBOUNDS = os.environ.get("PBHBOUNDS", os.path.join(_ext, "PBHbounds"))
CEFFYL = os.environ.get("CEFFYL", os.path.join(_ext, "ceffyl"))
NG15 = os.environ.get("NG15", os.path.join(_ext, "NG15"))
NPROC = int(os.environ.get("PBH_NPROC", max(1, (os.cpu_count() or 2) - 2)))
