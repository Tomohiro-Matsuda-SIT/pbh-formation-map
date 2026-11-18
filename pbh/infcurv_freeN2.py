"""Inflating curvaton with the n_s cut removed (KLM 2013 separate CMB generation from PBH formation) and N_2 up to
N2max; maximal f_PBH over a uniform scan of the box.  Optional: zeta_c."""
import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
os.chdir(r"C:\research\pbh\pbh"); sys.path.insert(0, os.getcwd())
import infcurv_pipeline as ic
N2max = float(sys.argv[1]); zc = float(sys.argv[2])
ic.BOX[4] = [1.0, N2max]
orig_bgi = ic.bgi
def bgi(key, N2):
    v = orig_bgi(key, N2)
    return np.full_like(np.asarray(N2, float), 0.97) if key == "ns" else v   # disable the n_s cut
ic.bgi = bgi
rng = np.random.default_rng(4)
P = ic.BOX[:, 0] + (ic.BOX[:, 1] - ic.BOX[:, 0]) * rng.random((1_500_000, 5))
ok = ic.constraints(P); A = P[ok]
print("allowed:", len(A), f"{ok.mean():.2e}", flush=True)
A = A[rng.choice(len(A), size=min(len(A), 30000), replace=False)]
res = {}
for mode in ("opt", "con"):
    lM, lf = ic.pbh_map(A, zeta_c=zc, skip=-1e9, mode=mode)
    fin = np.isfinite(lf)
    j = np.nanargmax(np.where(fin, lf, -np.inf))
    D = ic.derived(A[j:j+1])
    res[mode] = dict(max_logf=float(lf[j]), logM=float(lM[j]), N2=float(D["N2"][0]), g2=float(D["g2"][0]),
                     Nst=float(D["Nst"][0]), frac_gt_m12=float((lf > -12).mean()))
    print(mode, res[mode], flush=True)
json.dump(res, open(os.path.join(os.path.dirname(__file__), f"infcurv_free_{N2max}_{zc}.json"), "w"), indent=1)
