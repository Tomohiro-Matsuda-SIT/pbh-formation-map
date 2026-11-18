r"""
Axion-like curvaton: dependence on the collapse threshold zeta_c (Kohri-consistency review D1, 2026-11-16).

The reference maps use zeta_c = 1.0 (a placeholder, axion_pipeline.py).  The radiation-era threshold of
Harada, Yoo, Kohri (arXiv:1309.4201), delta_c ~ 0.41 for the density contrast at horizon crossing, corresponds to a
lower curvature threshold; we bracket it with zeta_c = 0.67 and 0.5.
Prior: the stored 3e5 samples (axion_prior_samples.npz) plus 7e5 new draws from the same truncated Gaussian
(axion_prior.npz), N = 1e6 in total.  Bounds: monochromatic current/projected PBHbounds (as build_results.py).
Stores conditions axion_curvaton/<mode>_zc<zc> and writes axion_zc.json.
usage: python axion_zc.py
"""
import os, sys, json, numpy as np, warnings
from concurrent.futures import ProcessPoolExecutor
from paths import PBH_ROOT, NPROC

NEW, NCH = 700_000, 14


def _draw(i):
    warnings.filterwarnings("ignore"); os.environ["OMP_NUM_THREADS"] = "1"
    import axion_pipeline as ap
    ap.rng = np.random.default_rng(91_000 + i)
    d = np.load(os.path.join(PBH_ROOT, "axion_prior.npz"))
    return ap.sample_prior(d["mean"], d["cov"], NEW // NCH)


def _map(args):
    P, zc, mode = args
    warnings.filterwarnings("ignore"); os.environ["OMP_NUM_THREADS"] = "1"
    import axion_pipeline as ap
    return ap.pbh_map(P, zeta_c=zc, skip=-40, mode=mode)


if __name__ == "__main__":
    os.chdir(PBH_ROOT); sys.path.insert(0, PBH_ROOT)
    fn = os.path.join(PBH_ROOT, "axion_prior_samples_1e6.npz")
    with ProcessPoolExecutor(NPROC) as ex:
        if os.path.exists(fn):
            P = np.load(fn)["P"]
        else:
            P = np.concatenate([np.load("axion_prior_samples.npz")["P"]] + list(ex.map(_draw, range(NCH))))
            np.savez(fn, P=P)
        N = len(P)
        edges = np.linspace(0, N, 8 * NPROC + 1).astype(int)
        import results_store as rs
        import emf

        def mono_okdet(lM, lf):          # as build_results.mono_okdet (importing that module would rebuild everything)
            cur = emf.mono_flags(lM, lf, emf.CURRENT); fut = emf.mono_flags(lM, lf, emf.FUTURE)
            return ~np.any(np.array(list(cur.values())), axis=0), np.any(np.array(list(fut.values())), axis=0)
        out = {}
        for zc in (1.0, 0.67, 0.5):
            for mode in ("opt", "con"):
                parts = list(ex.map(_map, [(P[a:b], zc, mode) for a, b in zip(edges[:-1], edges[1:])]))
                lM = np.concatenate([p[0] for p in parts]); lf = np.concatenate([p[1] for p in parts])
                ok, det = mono_okdet(lM, lf)
                g = ok & (lf > -12) & (lf < 0)
                ix = np.floor(lM[g] * 3).astype(int); iy = np.floor(lf[g] * 3).astype(int)
                cnt = np.unique(np.stack([ix, iy]), axis=1, return_counts=True)[1] if g.any() else np.array([0])
                st = dict(N=N, P_reach=float(g.mean()), max_cell=float(cnt.max() / N),
                          n_cells_ge_1e5=int((cnt / N >= 1e-5).sum()), P_excluded=float(np.mean((lf > -12) & ~ok)),
                          lM_5_95=[float(x) for x in np.percentile(lM[g], [5, 95])] if g.any() else None)
                out[f"{mode}_zc{zc}"] = st
                print(zc, mode, st, flush=True)
                rs.save("axion_curvaton", f"{mode}_zc{zc}", lM, lf, ok, det,
                        f"axion-like (KKY) curvaton, hilltop treatment '{mode}', collapse threshold zeta_c = {zc}; "
                        f"N = {N} prior samples", ["anharmonic G(theta) with hilltop asymptotics", "monochromatic MF"],
                        ["axion_G.py", "axion_pipeline.py", "axion_zc.py"], theta=P)
    json.dump(out, open(os.path.join(PBH_ROOT, "axion_zc.json"), "w"), indent=1)
