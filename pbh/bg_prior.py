r"""
Hierarchical prior on the background psi = (log10 alpha, w_reh, log10 T_reh) for the E- and T-models
(background.py), per CMB band (Planck / ACT), following the project rule:
  allowed region (CMB n_s 2 sigma, r < 0.036, BBN, rho_reh <= rho_end) inside the box
  -> Gaussian with the mean and covariance of the allowed region, truncated to it.
Box sampling is uniform in (log10 alpha, w, log10 T_reh) with log10 T_reh in [log10 5e-3, 16]; points with
rho_reh > rho_end are outside the box (so the upper T_reh edge is the instant-reheating line).
Output: bg_prior.npz with the box scan, allowed flags, Gaussian parameters and K background nodes per
(band, model) drawn from the truncated Gaussian (the nodes used to rerun the PBH models).
"""
import numpy as np, warnings, time
from multiprocessing import Pool
import background as bgm
from paths import PBH_ROOT, NPROC
warnings.filterwarnings("ignore")

NBOX = 6000            # box points per model
KNODE = 240           # background nodes per (band, model)
LT_HI = 16.0
rng = np.random.default_rng(20261115)

def _eval(args):
    model, la, w, lT = args
    b = bgm.solve(model, 10**la, w, 10**lT)
    if b is None:
        return (np.nan,) * 4
    return (b["ns"], b["r"], b["Nst"], b["alpha_s"])

def scan(model, n):
    X = np.column_stack([rng.uniform(*bgm.BOX["la"], n), rng.uniform(*bgm.BOX["w"], n),
                         rng.uniform(bgm.LT_MIN, LT_HI, n)])
    with Pool(NPROC) as p:
        R = np.array(p.map(_eval, [(model, *x) for x in X], chunksize=50))
    return X, R

def allowed(R, band):
    mu, sd = bgm.NS[band]
    return np.isfinite(R[:, 0]) & (np.abs(R[:, 0] - mu) <= 2 * sd) & (R[:, 1] < bgm.R_MAX)

def draw(mean, cov, model, band, k):
    out = []
    while len(out) < k:
        x = rng.multivariate_normal(mean, cov)
        if not (bgm.BOX["la"][0] <= x[0] <= bgm.BOX["la"][1] and bgm.BOX["w"][0] <= x[1] <= bgm.BOX["w"][1]
                and bgm.LT_MIN <= x[2] <= LT_HI):
            continue
        b = bgm.solve(model, 10**x[0], x[1], 10**x[2])
        if bgm.cmb_ok(b, band):
            out.append(x)
    return np.array(out)

if __name__ == "__main__":
    t = time.time(); res = {}
    for model in ("E", "T"):
        X, R = scan(model, NBOX)
        inbox = np.isfinite(R[:, 0])
        res[f"X_{model}"], res[f"R_{model}"] = X, R
        print(f"[{model}] box points {NBOX}, inside instant-reheating line {inbox.mean():.3f}  ({time.time()-t:.0f}s)")
        for band in ("planck", "act"):
            ok = allowed(R, band)
            res[f"ok_{model}_{band}"] = ok
            print(f"  {band}: allowed fraction of box {ok.sum() / inbox.sum():.4f} ({ok.sum()} pts)")
            if ok.sum() < 10:
                print("   -> (almost) empty allowed region"); continue
            A = X[ok]; mean, cov = A.mean(0), np.cov(A.T)
            print(f"     allowed ranges: log a [{A[:,0].min():.2f},{A[:,0].max():.2f}]  w [{A[:,1].min():.3f},{A[:,1].max():.3f}]"
                  f"  log T_reh [{A[:,2].min():.2f},{A[:,2].max():.2f}]   N_* [{R[ok,2].min():.1f},{R[ok,2].max():.1f}]")
            print(f"     mean {np.round(mean, 3)}  std {np.round(np.sqrt(np.diag(cov)), 3)}")
            print(f"     corr\n{np.round(np.corrcoef(A.T), 2)}")
            res[f"mean_{model}_{band}"], res[f"cov_{model}_{band}"] = mean, cov
            res[f"nodes_{model}_{band}"] = draw(mean, cov, model, band, KNODE)
    np.savez(PBH_ROOT + "/bg_prior.npz", **res)
    print(f"saved bg_prior.npz  ({time.time()-t:.0f}s)")
