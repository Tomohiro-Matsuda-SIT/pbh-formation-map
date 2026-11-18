"""Sensitivity of the curvaton-family reach to the collapse threshold zeta_c (1.0 placeholder vs lower values
motivated by Harada-Yoo-Kohri 2013 delta_c ~ 0.41).  Upper estimate: PBH bounds not applied."""
import os, sys, time, json, numpy as np, warnings
warnings.filterwarnings("ignore")
os.chdir(r"C:\research\pbh\pbh"); sys.path.insert(0, os.getcwd())
import axion_pipeline as ap
which = sys.argv[1]
out = {}
if which == "axion":
    P = np.load("axion_prior_samples.npz")["P"]
    N = len(P)
    for zc in (1.0, 0.8, 0.67, 0.5):
        for mode in ("opt", "con"):
            t = time.time()
            lM, lf = ap.pbh_map(P, zeta_c=zc, skip=-30, mode=mode)
            g = (lf > -12) & (lf < 0)
            # 1/3-dex cells in (log M/Msun, log f)
            ix = np.floor(lM[g] * 3).astype(int); iy = np.floor(lf[g] * 3).astype(int)
            _, cnt = np.unique(np.stack([ix, iy]), axis=1, return_counts=True)
            out[f"{zc}_{mode}"] = dict(P_reach=float(g.mean()), max_cell=float(cnt.max() / N) if len(cnt) else 0.0,
                                      n_cells_ge_eps=int((cnt / N >= 1e-5).sum()), f_gt1=float((lf > 0).mean()),
                                      lM_range=[float(np.nanmin(lM[g])), float(np.nanmax(lM[g]))] if g.any() else None)
            print(zc, mode, out[f"{zc}_{mode}"], f"{time.time()-t:.0f}s", flush=True)
else:
    import infcurv_pipeline as ic
    A = np.load("infcurv_allowed.npy") if os.path.exists("infcurv_allowed.npy") else ic.build_prior()[3]
    A = A[np.random.default_rng(1).choice(len(A), size=min(len(A), 40000), replace=False)]
    for zc in (1.0, 0.67, 0.5):
        for mode in ("opt", "con"):
            lM, lf = ic.pbh_map(A, zeta_c=zc, skip=-1e9, mode=mode)
            fin = np.isfinite(lf)
            out[f"{zc}_{mode}"] = dict(n=len(A), max_logf=float(lf[fin].max()) if fin.any() else None,
                                      n_gt_m12=int((lf > -12).sum()))
            print(zc, mode, out[f"{zc}_{mode}"], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), f"zc_{which}.json"), "w"), indent=1)
