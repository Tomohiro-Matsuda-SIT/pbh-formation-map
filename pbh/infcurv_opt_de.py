r"""
Inflating curvaton: global maximization of the PBH abundance over the allowed region with differential evolution
(verification of the null result, 2026-11-16).  Earlier checks used a local optimizer (16 starts) and uniform
scans (797 / 3e4 points), which can miss a thin feasible region.
Settings (argv[1]):
  ref    : constraints as in the paper (n_s > 0.9565, N_2 in [1, 8])
  freeN2 : n_s constraint removed, N_2 in [1, 29.9] (KLM 2013 separate the CMB from PBH formation)
  noC3a  : as freeN2, and additionally the classicality condition at the end of primordial inflation removed
           (KLM do not impose it)
  noMP   : as freeN2, and sigma_* < M_P / spectator-energy conditions removed (KLM impose neither)
zeta_c = argv[2] (default 1.0).  Objective: max log10 f_PBH (optimistic treatment), penalty if not allowed.
"""
import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.optimize import differential_evolution
import infcurv_pipeline as ic
import curvaton_pipeline as cp

mode = sys.argv[1] if len(sys.argv) > 1 else "ref"
zc = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
INPLANE = len(sys.argv) > 3 and sys.argv[3] == "inplane"   # require peak mass above 5e14 g (unevaporated)
BOX = ic.BOX.copy()
if mode != "ref":
    BOX[4] = [1.0, 29.9]
    _bgi = ic.bgi
    ic.bgi = lambda key, N2: np.full_like(np.asarray(N2, float), 0.97) if key == "ns" else _bgi(key, N2)
if mode == "noMP":
    BOX[1] = [3.0, 7.0]
if mode == "lowH":
    # KLM-like test: primordial inflation at H ~ 2.4e11 GeV (M_1 ~ 1e15 GeV), CMB from another source.
    # The Starobinsky H(N) is rescaled by S (N_* kept); the curvaton must still contribute < 1% of A_s on CMB scales.
    S = 2.4e11 / 1.6e13
    _b2 = ic.bgi
    ic.bgi = lambda key, N2: _b2(key, N2) * S if key in ("mu", "Hst", "Hend") else _b2(key, N2)
    BOX[1] = [3.0, 7.0]


def allowed(P):
    d = ic.derived(P)
    lin = (d["r"] / d["g2"])**2
    Pz_cmb = lin * d["Px"][:, -1]
    w = Pz_cmb / (ic.As + Pz_cmb)
    fNL = 5 / 6 * d["g2"] / d["r"] * w**2
    C1 = (Pz_cmb < 0.01 * ic.As) & (np.abs(fNL) < 5) & (d["ns"] > 0.9565)
    sst = 10**d["lsig"] * d["Hst"]
    C2 = (0.5 * d["c1"] * sst**2 < 0.03 * ic.MP**2) & (sst < ic.MP)
    if mode == "noMP":
        C2 = np.ones_like(C2)
    C3a = d["Px"][:, 0] < 1
    if mode == "noC3a":
        C3a = np.ones_like(C3a)
    C3b = d["H2"] / (2 * np.pi) < d["g2"] * np.exp(d["lnsini"])
    C4 = (d["T2"] > 5e-3) & (d["T2"] < ic.bgi("Treh", d["N2"])) & (d["eta2"] < 0.75) & (d["N2"] >= 1) \
        & (d["N2"] < 29.9) & (10**d["lsend"] <= np.sqrt(2 / d["eta2"]))
    lg = lambda a, b: np.maximum(np.log10(np.maximum(a, 1e-300) / b), 0)
    viol = lg(Pz_cmb, 0.01 * ic.As) + lg(np.abs(fNL), 5) + lg(0.9565, d["ns"]) * 100 \
        + (0 if mode == "noMP" else lg(0.5 * d["c1"] * sst**2, 0.03 * ic.MP**2) + lg(sst, ic.MP)) \
        + (0 if mode == "noC3a" else lg(d["Px"][:, 0], 1)) + lg(d["H2"] / (2 * np.pi), d["g2"] * np.exp(d["lnsini"])) \
        + lg(d["T2"], ic.bgi("Treh", d["N2"])) + lg(10**d["lsend"], np.sqrt(2 / d["eta2"]))
    allowed.viol = viol
    return C1 & C2 & C3a & C3b & C4, d


def obj(x):
    P = x[None, :]
    ok, d = allowed(P)
    if not ok[0]:
        return 1e9 + 1e6 * float(allowed.viol[0])
    lM, lf = ic.pbh_map(P, zeta_c=zc, skip=-1e12, mode="opt")
    if INPLANE and not (lM[0] > np.log10(5e14 / 1.98847e33)):
        return 1e7 - lM[0]                     # push towards heavier masses
    return -lf[0] if np.isfinite(lf[0]) else 1e8


best = None
for seed in range(4):
    r = differential_evolution(obj, BOX, seed=seed, popsize=40, maxiter=400, tol=1e-10, polish=False,
                               init="sobol", mutation=(0.5, 1.0), recombination=0.7)
    if best is None or r.fun < best.fun:
        best = r
    print(f"seed {seed}: max log10 f = {-r.fun:.4g}", flush=True)
ok, d = allowed(best.x[None, :])
lM, lf = ic.pbh_map(best.x[None, :], zeta_c=zc, skip=-1e12, mode="opt")
res = dict(mode=mode, zeta_c=zc, max_logf=float(lf[0]), logM=float(lM[0]), x=best.x.tolist(),
           g2=float(d["g2"][0]), N2=float(d["N2"][0]), Nst=float(d["Nst"][0]), c1=float(d["c1"][0]),
           Px_end=float(d["Px"][0, 0]), Px_cmb=float(d["Px"][0, -1]), sigma_star_over_MP=float(10**d["lsig"][0] * d["Hst"][0] / ic.MP))
print(res, flush=True)
json.dump(res, open(f"infcurv_de_{mode}_{zc}{'_inplane' if INPLANE else ''}.json", "w"), indent=1)
