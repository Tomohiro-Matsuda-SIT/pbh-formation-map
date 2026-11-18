r"""
Can the axion-like curvaton, on the reference inflation model and inside the allowed region R of axion_pipeline.py,
produce PBHs at the masses claimed in the literature?  (referee response: reconciliation with
Inomata et al. 2011.01270 [~30 Msun, NANOGrav] and Ferrante et al. 2305.13382 [asteroid mass, f = 1].)

For each target mass window  |log10 M/Msun - target| < 0.5  the abundance log10 f is maximised over the six
parameters by differential evolution, subject to the non-PBH conditions C1-C4 (constraints()).  The initial
population consists of allowed points, and a violated condition is penalised, so the search stays inside R.
A second run per target relaxes ONE condition at a time (C1 CMB, C2 light/spectator, C3 no hilltop crossing,
C4 decay) to identify which condition limits the abundance.
usage: python axion_target_de.py [zeta_c=1.0]       -> axion_target_de_<zc>.json
"""
import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.optimize import differential_evolution
import axion_pipeline as ap
import curvaton_pipeline as cp
from paths import PBH_ROOT

ZC = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
TARGETS = [-13.0, -9.0, -4.0, 0.0, 1.5]
BOX = ap.BOX; BG, MP = ap.BG, ap.MP

def conds(P):
    """the four conditions separately (same expressions as ap.constraints)."""
    d = ap.derived(P)
    r = 3 * d["Om"] / (4 - d["Om"])
    Pz = (r / 3 * d["dlg"])**2 * d["Pdth"][:, -1]
    w = Pz / (cp.As + Pz); fNL = 5 / (4 * np.maximum(r, 1e-300)) * w**2
    C1 = (Pz < 0.01 * cp.As) & (np.abs(fNL) < 5)
    phist = np.exp(d["lnphi"][:, -1])
    C2 = (d["m"] < 0.1 * BG["Hend"]) & (0.5 * d["c"] * phist**2 < 0.03 * MP**2) & (phist < MP)
    var = np.trapezoid(d["Pdth"][:, ::-1], ap.lnkN[::-1], axis=1)
    C3 = var < (np.pi - d["thi"])**2
    C4 = (d["Tdec"] > 5e-3) & (d["Tdec"] < BG["Treh"]) & (cp.H_rad(d["Tdec"]) < d["m"]**3 / (16 * np.pi * d["f"]**2))
    return np.array([C1, C2, C3, C4])

def run(target, mode, relax=None, seed=0, pop=120, gens=150):
    rng = np.random.default_rng(100 + seed)
    def feasible(P):
        C = conds(P)
        if relax is not None:
            C[relax] = True
        return C.all(0)
    # initial population: allowed points, preferring those with a mass near the target
    init = []
    while sum(len(a) for a in init) < 4 * pop:
        P = BOX[:, 0] + (BOX[:, 1] - BOX[:, 0]) * rng.random((200000, 6))
        init.append(P[feasible(P)])
    init = np.concatenate(init)[:4 * pop]
    lM0, lf0 = ap.pbh_map(init, zeta_c=ZC, skip=-1e9, mode=mode)
    sc = np.where(np.isfinite(lM0), np.abs(lM0 - target), 99)
    init = init[np.argsort(sc)[:pop]]
    def obj(X):
        P = np.atleast_2d(X.T if X.ndim == 2 else X)
        ok = feasible(P)
        lM, lf = ap.pbh_map(P, zeta_c=ZC, skip=-1e9, mode=mode)
        lf = np.where(np.isfinite(lf), np.maximum(lf, -1e5), -1e5)
        miss = np.where(np.isfinite(lM), np.maximum(np.abs(lM - target) - 0.5, 0), 50.0)
        val = -lf + 1e6 * miss          # the mass window dominates: any in-window point beats any point outside it
        return np.where(ok, val, 1e12)
    r = differential_evolution(obj, list(map(tuple, BOX)), init=init, maxiter=gens, tol=0, atol=0, seed=seed,
                               vectorized=True, updating="deferred", polish=False, mutation=(0.4, 1.0), recombination=0.8)
    x = r.x[None, :]
    lM, lf = ap.pbh_map(x, zeta_c=ZC, skip=-1e9, mode=mode)
    return dict(log10f=float(lf[0]), log10M=float(lM[0]), theta=r.x.tolist(), conds=conds(x)[:, 0].tolist(),
                in_window=bool(abs(lM[0] - target) <= 0.5))

if __name__ == "__main__":
    out = {"zeta_c": ZC, "params": ["c", "N_f", "log10 f/H_*", "theta_i", "log10 m/GeV", "log10 T_dec/GeV"]}
    for t in TARGETS:
        for mode in ("opt", "con"):
            best = max((run(t, mode, seed=s) for s in range(3)), key=lambda d: d["log10f"] if d["in_window"] else -1e9)
            if not best["in_window"]:
                best["log10f"] = None          # no allowed point found in the mass window
            out[f"M{t}_{mode}"] = best
            print(f"target {t:5.1f} [{mode}] in window: {best['in_window']}  max log10 f = {best['log10f']} at log10 M = {best['log10M']:.2f} "
                  f"theta = {np.round(best['theta'], 3).tolist()}", flush=True)
            if best["log10f"] is None or best["log10f"] < 0:
                for k, nm in enumerate(("C1", "C2", "C3", "C4")):
                    b = run(t, mode, relax=k, seed=0)
                    out[f"M{t}_{mode}_no{nm}"] = b
                    print(f"      without {nm}: max log10 f = {b['log10f']:.4g} at log10 M = {b['log10M']:.2f}", flush=True)
            json.dump(out, open(PBH_ROOT + f"/axion_target_de_{ZC}.json", "w"), indent=1)
