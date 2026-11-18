r"""
Inflation models of the alpha-attractor family that agree with the ACT DR6 value of n_s (referee response).
The coarse scan of bg_prior.py (6000 box points per model) finds only 9 ACT-allowed points, in two separate
islands of the E-model.  Here each island is scanned densely and treated as its own case:
    actH  large alpha (log10 alpha ~ 1.1-1.3), w ~ 0, T_reh ~ 1e14-1e15 GeV (close to instant reheating)
    actL  w ~ 0.5 (stiff reheating), T_reh ~ 5 MeV - 0.1 GeV
For each island and model (E, T): uniform scan of the island box, allowed region = ACT n_s (2 sigma), r < 0.036,
rho_reh <= rho_end; Gaussian with the moments of the allowed region truncated to it (project rule); K nodes.
Output: bg_prior_act.npz with nodes_<model>_<island>  (read by bg_run.py when PBH_BGPRIOR points to it).
usage: python bg_prior_act.py [K=60]
"""
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
import background as bgm
from paths import PBH_ROOT

K = int(sys.argv[1]) if len(sys.argv) > 1 else 60
NSCAN = 60000
ISL = {"actH": np.array([[0.5, 2.0], [0.0, 0.15], [12.0, 16.0]]),
       "actL": np.array([[-2.0, 2.0], [0.35, 0.5], [np.log10(5e-3), 1.0]])}
rng = np.random.default_rng(20261116)

def ok_act(model, x):
    b = bgm.solve(model, 10**x[0], x[1], 10**x[2])
    return b is not None and bool(bgm.cmb_ok(b, "act")), b

res = {}
for isl, box in ISL.items():
    for model in ("E", "T"):
        X = box[:, 0] + (box[:, 1] - box[:, 0]) * rng.random((NSCAN, 3))
        fl, R = [], []
        for x in X:
            o, b = ok_act(model, x); fl.append(o); R.append((b["ns"], b["r"], b["Nst"]) if o else (np.nan,) * 3)
        fl = np.array(fl); R = np.array(R); A = X[fl]
        print(f"[{isl}/{model}] allowed {fl.sum()} of {NSCAN} ({fl.mean():.4f})")
        if fl.sum() < 50:
            print("   -> too few points, skipped"); continue
        mean, cov = A.mean(0), np.cov(A.T)
        print(f"   log a [{A[:,0].min():.2f},{A[:,0].max():.2f}]  w [{A[:,1].min():.3f},{A[:,1].max():.3f}]  "
              f"log T_reh [{A[:,2].min():.2f},{A[:,2].max():.2f}]  n_s [{np.nanmin(R[:,0]):.4f},{np.nanmax(R[:,0]):.4f}]  "
              f"r [{np.nanmin(R[:,1]):.4f},{np.nanmax(R[:,1]):.4f}]  N_* [{np.nanmin(R[:,2]):.1f},{np.nanmax(R[:,2]):.1f}]")
        print(f"   mean {np.round(mean, 3)}  std {np.round(np.sqrt(np.diag(cov)), 3)}")
        nodes = []
        while len(nodes) < K:
            x = rng.multivariate_normal(mean, cov)
            if np.all((x >= box[:, 0]) & (x <= box[:, 1])) and ok_act(model, x)[0]:
                nodes.append(x)
        res[f"nodes_{model}_{isl}"] = np.array(nodes)
        res[f"allowed_{model}_{isl}"] = A; res[f"R_{model}_{isl}"] = R[fl]
        res[f"frac_{model}_{isl}"] = fl.mean() * np.prod(box[:, 1] - box[:, 0])     # allowed volume in (la, w, lT)
np.savez(PBH_ROOT + "/bg_prior_act.npz", **res)
print("saved bg_prior_act.npz:", [k for k in res if k.startswith("nodes")])
