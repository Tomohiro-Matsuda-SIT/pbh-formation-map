"""
Rebuild the stored results from the inputs in this repository, then verify against expected/.

usage (from the pbh/ directory, venv active):
    python regenerate.py --check            # environment + data check only (seconds)
    python regenerate.py                    # standard rebuild (uses the shipped usr_nodes.npy)
    python regenerate.py --full             # also redo the slow scans (USR node scan, inflationary walls,
                                            # cosmic strings) instead of using the shipped results
    python regenerate.py --from 9           # resume from step 9
    python regenerate.py --only 14,15       # run selected steps
    python regenerate.py --tex              # additionally compile docs/pbh_models_explained.tex with lualatex

Each step runs as a subprocess in this directory with PYTHONUTF8=1; logs go to logs/step_XX.log.
Timings in the comments are rough estimates for an 8-core / 16-thread desktop (Ryzen 7 8700G, 32 GB).
"""
import os, sys, json, shutil, subprocess, time, argparse, glob

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
PY = sys.executable

STEPS = [
    # (id, description, argv, full_only)
    (1, "PTA grid fit, NANOGrav 15-yr (DW)", ["dw_pta_fit.py", "NG15"], False),            # ~2 min
    (2, "PTA grid fit, EPTA DR2new (DW)", ["dw_pta_fit.py", "EPTA"], False),               # ~1 min
    (3, "FOPT + DW priors and samples -> fopt_dw_samples.npz (~1.3 GB)", ["fopt_dw_pipeline.py"], False),  # 20-40 min
    (4, "DW mass-function width (critical collapse) for cases ul/pta", ["patch_dw_width.py"], False),      # 10-20 min
    (5, "DW explains PTA, EPTA likelihood (case ptaLE)", ["run_dw_ptaL.py", "ptaLE"], False),               # ~10 min
    (6, "DW explains PTA, NG15 likelihood (case ptaL)", ["run_dw_ptaL.py", "ptaL"], False),                 # ~10 min
    (7, "DW, PTA as upper limit with SMBHB, NG15 (case ulL)", ["run_dw_ptaL.py", "ulL"], False),            # ~10 min
    (8, "inflationary walls (slow)", ["infwall_pipeline.py"], True),                                         # ~30 min
    (9, "cosmic strings", ["cosmicstring_pipeline.py"], True),                                               # ~5 min
    (10, "result store: all models x conditions (FOPT and Fermi-ball are recomputed here)", ["build_results.py"], False),  # 20-40 min
    (11, "USR node scan (55 nodes, parallel; shipped usr_nodes.npy is used unless --full)", ["usr_pipeline.py"], True),    # ~15 min at 14 procs
    (12, "USR pushforward (conditional integration)", ["usr_pushforward.py"], False),                       # ~2 min
    (13, "USR prior sensitivity", ["usr_prior_sens.py"], False),                                             # ~2 min
    (14, "re-plot every individual map", ["results_store.py"], False),
    (15, "combined (A)/(B) figure, main", ["fig_combined.py"], False),
    (16, "combined figure variants", ["fig_combined.py", "mono"], False),
    (17, "combined figure variants", ["fig_combined.py", "epta"], False),
    (18, "combined figure variants", ["fig_combined.py", "legacy"], False),
    (19, "gallery PDF", ["make_gallery.py"], False),
]

def check():
    ok = True
    print(f"python {sys.version.split()[0]} at {PY}")
    try:
        import numpy, scipy, matplotlib
        print(f"numpy {numpy.__version__}, scipy {scipy.__version__}, matplotlib {matplotlib.__version__}")
        if int(numpy.__version__.split(".")[0]) < 2:
            print("  !! numpy >= 2.0 required (np.trapezoid)"); ok = False
    except ImportError as e:
        print("  !! missing package:", e); ok = False
    sys.path.insert(0, HERE)
    from paths import PBHBOUNDS, CEFFYL, NG15, NPROC
    need = [(os.path.join(PBHBOUNDS, "bounds", "Evaporation.txt"), "PBHbounds"),
            (os.path.join(CEFFYL, "data", "epta_dr2_new_hd", "density.npy"), "ceffyl EPTA"),
            (os.path.join(NG15, "30f_fs{hd}_ceffyl", "density.npy"), "NANOGrav 15-yr KDE (unzipped)")]
    for p, nm in need:
        print(f"  {'ok ' if os.path.exists(p) else '!! MISSING'} {nm}: {p}"); ok &= os.path.exists(p)
    for f in ["samples.npz", "axion_prior_samples.npz", "infcurv_scan.npz", "infwall_samples.npz",
              "cstring_samples.npz", "usr_nodes.npy"]:
        p = os.path.join(HERE, f); print(f"  {'ok ' if os.path.exists(p) else '!! MISSING'} shipped data {f}"); ok &= os.path.exists(p)
    print(f"  NPROC = {NPROC}")
    return ok

def run(step):
    sid, desc, argv, _ = step
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    log = os.path.join(HERE, "logs", f"step_{sid:02d}.log")
    env = dict(os.environ, PYTHONUTF8="1", MPLBACKEND="Agg")
    print(f"[{time.strftime('%H:%M:%S')}] step {sid:2d}: {desc}  ({' '.join(argv)}) -> {os.path.relpath(log, HERE)}", flush=True)
    t = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        r = subprocess.run([PY, "-u"] + argv, cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT)
    print(f"           done in {(time.time() - t) / 60:.1f} min, exit {r.returncode}", flush=True)
    if r.returncode != 0:
        print(open(log, encoding="utf-8").read()[-3000:])
        sys.exit(f"step {sid} failed; fix and resume with --from {sid}")

def verify():
    """compare (A)/(B) fractions and per-result reach probabilities with the cloud-session values."""
    exp = os.path.join(BUNDLE, "expected")
    print("\n=== verification against expected/ (cloud session 2026-11-15) ===")
    bad = 0
    for fn in sorted(glob.glob(os.path.join(exp, "fig_combined_AB*_stats.json"))):
        new = os.path.join(HERE, os.path.basename(fn))
        if not os.path.exists(new):
            continue
        a, b = json.load(open(fn)), json.load(open(new))
        for k, v in a.items():
            if isinstance(v, dict) and "A" in v and k in b:
                d = max(abs(v[c] - b[k][c]) for c in ("A", "B1", "B2"))
                flag = "ok " if d <= 0.03 else "DIFF"
                bad += d > 0.03
                print(f"  {flag} {os.path.basename(fn):34s} {k:10s} expected {v}  got {b[k]}")
    ia = json.load(open(os.path.join(exp, "results_index.json")))
    ib = json.load(open(os.path.join(HERE, "results", "index.json")))
    for k, v in ia.items():
        if v.get("stats") is None or k not in ib or ib[k].get("stats") is None:
            continue
        pa, pb = v["stats"]["P_reach"], ib[k]["stats"]["P_reach"]
        rel = abs(pa - pb) / max(pa, 1e-12) if pa > 0 else abs(pb)
        tol = 0.15 if pa > 1e-3 else 1.0          # Monte-Carlo noise (fixed seeds, but call order may differ)
        flag = "ok " if rel <= tol else "DIFF"
        bad += rel > tol
        print(f"  {flag} {k:32s} P_reach expected {pa:.3g} got {pb:.3g}")
    print(f"=== {bad} differences beyond tolerance ===")
    return bad

def tex():
    docs = os.path.join(BUNDLE, "docs")
    shutil.copy(os.path.join(HERE, "fig_combined_AB.pdf"), docs)
    if shutil.which("lualatex") is None:
        print("lualatex not found on PATH; skip"); return
    for _ in range(2):
        subprocess.run(["lualatex", "-interaction=nonstopmode", "pbh_models_explained.tex"], cwd=docs)
    print("compiled docs/pbh_models_explained.pdf")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--tex", action="store_true")
    ap.add_argument("--verify", action="store_true", help="only run the verification")
    a = ap.parse_args()
    if not check():
        sys.exit("environment check failed (see above)")
    if a.check:
        sys.exit(0)
    if a.verify:
        sys.exit(1 if verify() else 0)
    only = {int(x) for x in a.only.split(",") if x.strip()}
    for st in STEPS:
        sid, _, argv, full_only = st
        if only and sid not in only:
            continue
        if not only and sid < a.start:
            continue
        if full_only and not a.full and not (only and sid in only):
            print(f"           skip step {sid:2d} ({st[1]}) -- shipped data used; pass --full to redo")
            continue
        if sid == 11 and os.path.exists(os.path.join(HERE, "usr_nodes.npy")):
            shutil.move(os.path.join(HERE, "usr_nodes.npy"), os.path.join(HERE, "usr_nodes_shipped.npy"))
        run(st)
    verify()
    if a.tex:
        tex()
