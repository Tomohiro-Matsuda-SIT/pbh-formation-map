r"""
Walls nucleated during inflation: sensitivity to walls that re-enter the horizon BEFORE reheating
(Kohri-consistency review C1, 2026-11-16).

Reference (infwall_pipeline, REH="drop"): such walls are discarded.  Alternative (REH="collapse"): they collapse during
the reheating era (constant w_reh), where pressure does not hinder collapse (Harada, Yoo, Kohri, Nakao, Jhingan,
arXiv:1609.01588); see infwall_pipeline._setup for the relations used.

usage:
  python infwall_reh.py node BG.json OUT.npz NSAMP      (one background; PBH_BG and PBH_IW_REH must be set by caller)
  python infwall_reh.py run [NSAMP=20000]               (all planck nodes + reference, both modes, in parallel)
  python infwall_reh.py combine                         (hierarchical statistics; stores conditions *_rehc_bgP)
Prior per node: identical to bg_node.py (same seed rule, nw = max(NSAMP//5, 5000)).
"""
import os, sys, glob, json, zlib, subprocess, numpy as np, warnings
warnings.filterwarnings("ignore")
from paths import PBH_ROOT, NPROC

BANDS = (("opt", 1e2, 2.0), ("cen", 1.0, 2.0), ("con", 1e-2, 1.0))
OUTD = os.path.join(PBH_ROOT, "bgscan", "planck_iwreh")


def node(out, ns):
    import curvaton_pipeline as cp
    import emf
    import infwall_pipeline as iw
    seed = zlib.crc32(os.path.basename(out).replace(f"_{iw.REH}", "").encode())
    iw.rng = np.random.default_rng(seed + 1)
    nw = max(ns // 5, 5000)
    res = {}
    for tag, A, kap in BANDS:
        if not iw.keep.any():
            break
        b = iw.box(A)
        u = b[0] + (b[1] - b[0]) * iw.rng.random(400_000)
        x = iw.rng.normal(u.mean(), u.std(), 6 * nw); x = x[(x >= b[0]) & (x <= b[1])][:nw]
        out_ = []
        for i in range(0, len(x), 1500):
            lM, psi, ftot, lMpk = iw.wall_mf(x[i:i + 1500], A, kap)
            logf = np.where(ftot > 0, np.log10(np.maximum(ftot, 1e-300)), -np.inf)
            sel = logf > -12
            ok = iw.wall_mf.evap_ok.copy(); det = np.zeros(len(logf), bool)
            if sel.any():
                Ic = emf.integrals(lM[sel], psi[sel], emf.CURRENT); If = emf.integrals(lM[sel], psi[sel], emf.FUTURE)
                ok[sel] &= np.all(np.array([Ic[k] <= 1 for k in Ic]), axis=0)
                det[sel] = np.any(np.array([If[k] > 1 for k in If]), axis=0)
            out_.append((lMpk, logf, ok, det))
        res[f"infwall__{tag}"] = np.vstack([np.asarray(np.concatenate(z), np.float32) for z in zip(*out_)])
    np.savez_compressed(out, bg=json.dumps({k: v for k, v in cp.BG.items()}, default=float), **res)


def run(ns):
    os.makedirs(OUTD, exist_ok=True)
    src = sorted(glob.glob(os.path.join(PBH_ROOT, "bgscan", "planck", "[ET]_[0-9]*.json")))
    jobs = [(p, m) for p in src for m in ("drop", "collapse")] + [(None, "drop"), (None, "collapse")]

    def one(job):
        p, mode = job
        name = (os.path.basename(p)[:-5] if p else "ref") + f"_{mode}.npz"
        out = os.path.join(OUTD, name)
        if os.path.exists(out):
            return
        env = dict(os.environ, PBH_IW_REH=mode, PYTHONUTF8="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        env.pop("PBH_BG", None)
        if p:
            env["PBH_BG"] = p
        r = subprocess.run([sys.executable, __file__, "node", out, str(ns)], env=env, capture_output=True,
                           text=True, cwd=PBH_ROOT)
        if r.returncode:
            print("FAILED", name, r.stderr[-1500:], flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(NPROC) as ex:
        list(ex.map(one, jobs))


def cells(lM, lf, w, ok):
    g = ok & (lf > -12) & (lf < 0) & np.isfinite(lM)
    ix = np.floor(lM[g] * 3).astype(int); iy = np.floor(lf[g] * 3).astype(int)
    if not g.any():
        return {}
    key = ix * 1000 + iy
    u, inv = np.unique(key, return_inverse=True)
    p = np.bincount(inv, weights=w[g])
    return dict(zip(u.tolist(), p.tolist()))


def combine():
    import results_store as rs
    out = {}
    for mode in ("drop", "collapse"):
        files = sorted(glob.glob(os.path.join(OUTD, f"[ET]_[0-9]*_{mode}.npz")))
        K = {m: sum(os.path.basename(f)[0] == m for f in files) for m in "ET"}
        for tag, _, _ in BANDS:
            L = {q: [] for q in ("lM", "lf", "ok", "det", "w", "lT")}
            for f in files:
                d = np.load(f); bg = json.loads(str(d["bg"]))
                k = f"infwall__{tag}"
                if k not in d.files:
                    continue
                a = d[k]; m = a.shape[1]
                L["lM"].append(a[0]); L["lf"].append(a[1]); L["ok"].append(a[2] > 0); L["det"].append(a[3] > 0)
                L["w"].append(np.full(m, 1.0 / (2 * K[os.path.basename(f)[0]] * m)))
                L["lT"].append(np.full(m, np.log10(bg["Treh"])))
            A = {q: np.concatenate(v) for q, v in L.items()}
            reach = A["ok"] & (A["lf"] > -12) & (A["lf"] < 0)
            win = reach & (A["lM"] > -16) & (A["lM"] < -10)
            c = cells(A["lM"], A["lf"], A["w"], A["ok"])
            o = np.argsort(A["lM"][reach]); cw = np.cumsum(A["w"][reach][o])
            med = float(A["lM"][reach][o][np.searchsorted(cw, cw[-1] / 2)]) if reach.any() else None
            st = dict(P_reach=float(A["w"][reach].sum()), P_window=float(A["w"][win].sum()), median_logM=med,
                      max_cell=max(c.values()) if c else 0.0, n_cells_ge_1e5=int(sum(v >= 1e-5 for v in c.values())),
                      cells_ge_1e5=sorted(k for k, v in c.items() if v >= 1e-5))
            out[f"bgP_{tag}_{mode}"] = st
            print(mode, tag, {k: v for k, v in st.items() if k != "cells_ge_1e5"}, flush=True)
            if mode == "collapse":
                rs.save("infwall", f"{tag}_rehc_bgP", A["lM"].astype(float), A["lf"].astype(float), A["ok"], A["det"],
                        "walls nucleated during inflation, walls re-entering before reheating also collapse "
                        "(reheating era with constant w_reh); alpha-attractor background family, planck band",
                        ["infwall_pipeline._setup(REH='collapse')", "per-node prior as bg_node.py"],
                        ["infwall_pipeline.py", "infwall_reh.py"], extra=dict(w=A["w"], lTreh=A["lT"]))
        for f in glob.glob(os.path.join(OUTD, f"ref_{mode}.npz")):
            d = np.load(f)
            for tag, _, _ in BANDS:
                a = d[f"infwall__{tag}"]; ok = a[2] > 0; reach = ok & (a[1] > -12) & (a[1] < 0)
                c = cells(a[0], a[1], np.full(a.shape[1], 1.0 / a.shape[1]), ok)
                out[f"ref_{tag}_{mode}"] = dict(P_reach=float(reach.mean()),
                                                median_logM=float(np.median(a[0][reach])) if reach.any() else None,
                                                max_cell=max(c.values()) if c else 0.0,
                                                P_evap_excluded=float(np.mean(~ok)))
                print("ref", mode, tag, out[f"ref_{tag}_{mode}"], flush=True)
    json.dump(out, open(os.path.join(PBH_ROOT, "infwall_reh.json"), "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "node":
        node(sys.argv[2], int(sys.argv[3]))
    elif cmd == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 20_000)
    else:
        combine()
