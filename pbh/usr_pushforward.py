r"""
Pushforward of the prior for the USR (Starobinsky + Gaussian bump) channel, using the node scan usr_nodes.npy.

Prior (project rule): Gaussian with the mean and covariance of the ALLOWED region, truncated to it, in
    theta = (phi0, log10 s, log10 A),  box phi0 in [2.9, 5.1], log10 s in [-2.3, -1.3], log10 A in [-6, -1].
Allowed region (non-PBH requirements), separately for the two CMB conditions ('planck', 'act'):
    A < A_c (not trapped);  CMB n_s, alpha_s (2 sigma) and r < 0.036;  mu < 9e-5;  SIGW: NG15 likelihood and
    LVK O3;  classicality proxy P_max < 1.
    Far from the critical height (delta = 1 - A/A_c above the scanned range, P_max < 1e-4) the spectrum is
    featureless and the CMB values of the largest-delta scan point are used.
Exact conditional integration: the PBH-relevant set is an extremely thin shell in log A just below A_c, which
random sampling would never hit.  For every node (phi0, s) [cell 0.2 x 0.2 in (phi0, log10 s)] the prior
probability of a delta-interval is p(theta) * Delta log10 A * cell area; within an interval log f and log M are
interpolated linearly in log delta on a sub-grid of NSUB points (ok/det taken from the nearer scan point).
The resulting weighted samples (lM, lf, ok, det, w) are stored per condition in results/usr_bump/<cmb>_<ng>/.
Fine-tuning diagnostics per node: Delta ln A of the shell with 1e-12 < f < 1, and the Barbieri-Giudice measure
|d ln f / d ln A| at f = 1e-3.
"""
import numpy as np, json, os
import results_store as rs
from usr_pipeline import cmb_ok

BOX = np.array([[2.9, 5.1], [-2.3, -1.3], [-6.0, -1.0]])
NSUB = 40
rng = np.random.default_rng(11)
# PBH_USR_NODES / PBH_USR_TAG: alternative node file (finer grid in the bump position) and suffix of the stored result
SUF = os.environ.get("PBH_USR_TAG", "")
nodes = list(np.load(os.environ.get("PBH_USR_NODES", "usr_nodes.npy"), allow_pickle=True))
phis = np.unique(np.round([n["phi0"] for n in nodes], 4)); lss = np.unique(np.round(np.log10([n["s"] for n in nodes]), 3))
dph, dls = phis[1] - phis[0], lss[1] - lss[0]

def node_arrays(n, which):
    R = n["recs"]
    d = np.array([r["delta"] for r in R])
    okc = np.array([cmb_ok(r["ns"], r["als"], r["r"], which) and r["mu"] < 9e-5 and r["pta_ok"] and r["lvk_ok"]
                    and r["Pmax"] < 1 for r in R])
    o = np.argsort(d)[::-1]          # decreasing delta (increasing A)
    return d[o], okc[o], [R[i] for i in o]

def allowed_logA_max(n, which):
    """largest log10 A such that every A below it (down to the box edge) is allowed; None if far region fails."""
    d, okc, R = node_arrays(n, which)
    far = cmb_ok(R[0]["ns"], R[0]["als"], R[0]["r"], which)
    if not far or not okc[0]:
        return None
    j = np.argmax(~okc) if (~okc).any() else len(okc)
    dlim = d[j - 1] if j < len(okc) else d[-1]
    return np.log10(n["Ac"] * (1 - dlim))

def gaussian_from_allowed(which, nbox=300_000):
    grid_lA = {}
    for n in nodes:
        grid_lA[(round(n["phi0"], 4), round(np.log10(n["s"]), 3))] = allowed_logA_max(n, which)
    P = BOX[:, 0] + (BOX[:, 1] - BOX[:, 0]) * rng.random((nbox, 3))
    ip = np.clip(np.round((P[:, 0] - phis[0]) / dph).astype(int), 0, len(phis) - 1)
    il = np.clip(np.round((P[:, 1] - lss[0]) / dls).astype(int), 0, len(lss) - 1)
    lim = np.array([grid_lA.get((round(phis[a], 4), round(lss[b], 3))) for a, b in zip(ip, il)], dtype=object)
    lim = np.array([-np.inf if v is None else v for v in lim], float)
    A = P[P[:, 2] < lim]
    if len(A) < 100:
        return None
    mu, cov = A.mean(0), np.cov(A.T)
    # normalisation of the truncated Gaussian over the allowed region (MC)
    Q = rng.multivariate_normal(mu, cov, size=1_000_000)
    inb = np.all((Q >= BOX[:, 0]) & (Q <= BOX[:, 1]), axis=1)
    iq = np.clip(np.round((Q[:, 0] - phis[0]) / dph).astype(int), 0, len(phis) - 1)
    jq = np.clip(np.round((Q[:, 1] - lss[0]) / dls).astype(int), 0, len(lss) - 1)
    limq = np.array([grid_lA.get((round(phis[a], 4), round(lss[b], 3))) for a, b in zip(iq, jq)], dtype=object)
    limq = np.array([-np.inf if v is None else v for v in limq], float)
    Z = np.mean(inb & (Q[:, 2] < limq))
    icov = np.linalg.inv(cov); det = np.linalg.det(cov)
    pdf = lambda th: np.exp(-0.5 * np.einsum("...i,ij,...j", th - mu, icov, th - mu)) / np.sqrt((2 * np.pi)**3 * det) / Z
    return dict(mu=mu, cov=cov, Z=Z, frac_allowed=len(A) / nbox, pdf=pdf, lim=grid_lA)

def pushforward(which, ng, G):
    lM_all, lf_all, ok_all, det_all, w_all = [], [], [], [], []
    diag = []
    for n in nodes:
        d, okc, R = node_arrays(n, which)
        lAmax = G["lim"].get((round(n["phi0"], 4), round(np.log10(n["s"]), 3)))
        lf = np.log10(np.maximum([r[f"fnow_{ng}"] for r in R], 1e-300))
        lM = np.array([r[f"lMpk_{ng}"] for r in R])
        ok = np.array([r[f"ok_{ng}"] for r in R]); det = np.array([r[f"det_{ng}"] for r in R])
        ld = np.log10(d)
        # fine-tuning diagnostics (independent of the prior)
        m = np.isfinite(lM)
        if (lf > -12).any() and (lf < 0).any():
            lA = np.log10(n["Ac"] * (1 - d))
            o = np.argsort(lf)
            try:
                a12, a0 = np.interp([-12, 0], lf[o], lA[o])
                f3 = np.interp(-3, lf[o], np.arange(len(lf))[o])
                i3 = int(round(f3)); i3 = min(max(i3, 1), len(lf) - 2)
                bg = abs((lf[i3 + 1] - lf[i3 - 1]) / (lA[i3 + 1] - lA[i3 - 1]))
                diag.append(dict(phi0=n["phi0"], s=n["s"], dlnA_shell=float(abs(a0 - a12) * np.log(10)), BG=float(bg)))
            except Exception:
                pass
        if lAmax is None:
            continue
        for j in range(len(R) - 1):
            if not (okc[j] and okc[j + 1]):
                continue
            if lf[j] < -14 and lf[j + 1] < -14:
                continue
            t = np.linspace(0, 1, NSUB, endpoint=False) + 0.5 / NSUB
            lds = ld[j] + t * (ld[j + 1] - ld[j])
            lfs = lf[j] + t * (lf[j + 1] - lf[j])
            lMa = lM[j] if np.isfinite(lM[j]) else lM[j + 1]
            lMb = lM[j + 1] if np.isfinite(lM[j + 1]) else lM[j]
            lMs = lMa + t * (lMb - lMa)
            near = np.where(t < 0.5, j, j + 1)
            ds = 10**lds
            lA_s = np.log10(n["Ac"] * (1 - ds))
            dlA = abs(np.log10(1 - d[j + 1]) - np.log10(1 - d[j])) / NSUB
            th = np.c_[np.full(NSUB, n["phi0"]), np.full(NSUB, np.log10(n["s"])), lA_s]
            w = G["pdf"](th) * dlA * dph * dls
            lM_all.append(lMs); lf_all.append(lfs); ok_all.append(ok[near]); det_all.append(det[near]); w_all.append(w)
    if not w_all:
        return None, diag
    cat = lambda L: np.concatenate(L)
    return (cat(lM_all), cat(lf_all), cat(ok_all), cat(det_all), cat(w_all)), diag

if __name__ == "__main__":
    summary = {}
    for which in ("planck", "act"):
        G = gaussian_from_allowed(which)
        for ng in ("G", "NG"):
            tag = f"{which}_{ng}{SUF}"
            desc = (f"inflaton-generated PBHs: Starobinsky + Gaussian bump (USR), CMB = {which}, "
                    f"{'Gaussian (smooth transition)' if ng == 'G' else 'USR exponential tail f_NL=5/2 (sharp transition)'}")
            assum = ["exact background + Mukhanov-Sasaki (usr_solver.py)", "compaction threshold (Musco+21), Ferrante+23 statistics",
                     "SIGW Kohri-Terada; NG15 likelihood + LVK", "mu < 9e-5", "P_max < 1 classicality proxy",
                     "prior: truncated Gaussian on allowed (phi0, log s, log A); conditional integration in log A"]
            if G is None:
                rs.save_note("usr_bump", tag, desc + " -- ALLOWED REGION EMPTY (base n_s outside the CMB band)", assum,
                             ["usr_solver.py", "usr_pbh.py", "usr_pipeline.py", "usr_pushforward.py"])
                summary[tag] = "allowed region empty"
                print(tag, "allowed region empty"); continue
            res, diag = pushforward(which, ng, G)
            if res is None:
                rs.save_note("usr_bump", tag, desc + " -- no PBH-producing point inside the allowed region", assum,
                             ["usr_solver.py", "usr_pbh.py", "usr_pipeline.py", "usr_pushforward.py"])
                summary[tag] = "no PBH in allowed region"; print(tag, summary[tag]); continue
            lM, lf, ok, det, w = res
            st = rs.save("usr_bump", tag, lM, lf, ok, det, desc, assum,
                         ["usr_solver.py", "usr_pbh.py", "usr_pipeline.py", "usr_pushforward.py"], extra=dict(w=w))
            summary[tag] = dict(stats=st, allowed_fraction_of_box=G["frac_allowed"], prior_mean=G["mu"].tolist(),
                                diag_median_dlnA=float(np.median([x["dlnA_shell"] for x in diag])) if diag else None,
                                diag_median_BG=float(np.median([x["BG"] for x in diag])) if diag else None)
            print(tag, json.dumps(summary[tag], indent=0)[:600])
    json.dump(summary, open(f"usr_summary{SUF}.json", "w"), indent=1, default=str)
