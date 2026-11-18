r"""
Inflaton-generated PBHs (Starobinsky + Gaussian bump, ultra-slow-roll) -- node scan.

For each node (phi0, sigma) of the feature:
  1. A_c = critical bump height above which the field is trapped (bisection on the exact background).
  2. delta = 1 - A/A_c sampled so that the peak power spans P_max ~ 3e-4 ... 0.3 (P_max ~ delta^-2 found
     numerically); MS spectrum at each point (usr_solver), then
       CMB: n_s, alpha_s, r at k_* = 0.05 Mpc^-1;  mu distortion;  SIGW (NANOGrav 15-yr likelihood as an
       upper limit with SMBHB, LVK O3 pointwise);  classicality proxy P_max < 1 (quantum diffusion
       dominates the USR plateau otherwise; 要確認, cf. stochastic-inflation literature);
       PBH mass function for the Gaussian ('G') and USR non-Gaussian ('NG') bands (usr_pbh) with current /
       projected bounds (emf.py) and evaporated-PBH limits (evap.py).
  3. Small-delta points are also kept at A << A_c (no feature effect) implicitly: for delta >~ 0.1 the spectrum
     is featureless (checked) and passes every constraint with f = 0.
Output: usr_nodes.npz (one record per node with the delta-scan arrays).
CMB bands (2 sigma): Planck 2018 [arXiv:1807.06211]: n_s = 0.9649 +- 0.0042, alpha_s = -0.0045 +- 0.0067;
                     P-ACT-LB [ACT DR6, arXiv:2503.14452 / 2503.14454]: n_s = 0.9743 +- 0.0034, alpha_s = 0.0062 +- 0.0052;
                     r_0.05 < 0.036 (BICEP/Keck 2021, arXiv:2110.00483).
"""
from paths import NPROC
import sys, numpy as np, warnings, time
warnings.filterwarnings("ignore")
import usr_solver as us, usr_pbh as up, emf, evap
import fopt_dw_pipeline as fd

BASE = "staro"
CMB = {"planck": dict(ns=(0.9649, 0.0042), als=(-0.0045, 0.0067)),
       "act": dict(ns=(0.9743, 0.0034), als=(0.0062, 0.0052))}
MSUN_LG = np.log10(1.98847e33)

def cmb_ok(ns, als, r, which):
    c = CMB[which]
    return (abs(ns - c["ns"][0]) < 2 * c["ns"][1]) & (abs(als - c["als"][0]) < 2 * c["als"][1]) & (r < 0.036)

def critical_A(phi0, s, base=BASE):
    lo, hi = 0.0, 0.2
    V, dV = us.make_potential(base, hi, phi0, s)
    while not us.background(V, dV, 6.5)[1]:
        hi *= 2
        V, dV = us.make_potential(base, hi, phi0, s)
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        V, dV = us.make_potential(base, mid, phi0, s)
        tr = us.background(V, dV, 6.5)[1]
        lo, hi = (mid, hi) if not tr else (lo, mid)
    return lo

def evaluate(phi0, s, A):
    o = us.spectrum(BASE, A, phi0, s, nk=220, dN=3e-3)
    if o["trapped"]:
        return None
    lnk, P = o["lnk"], o["P"]
    if not np.all(np.isfinite(P)) or not np.isfinite(o["ns"]):
        return None                                  # numerical overflow (P >> 1): treated like trapped
    i = int(np.argmax(P))
    rec = dict(ns=o["ns"], als=o["alpha_s"], r=o["r"], Nst=o["Nst"], Pmax=float(P[i]), lkp=float(lnk[i]),
               mu=up.mu_distortion(lnk, P))
    # SIGW
    kp = np.exp(lnk[i])
    kg = kp * np.exp(np.linspace(-4, 1.2, 40))
    fG, h2O = up.sigw(lnk, P, kg)
    Om = lambda f: np.exp(np.interp(np.log(f), np.log(fG), np.log(np.maximum(h2O, 1e-300)), left=-700, right=-700))[None, :] / fd.h**2
    rec["pta_ok"] = bool(fd.pta_ok_L(Om)[0])
    rec["lvk_ok"] = bool(np.all(Om(fd.f_LVK) < 5.8e-9))
    rec["sigw_peak"] = float(h2O.max()); rec["sigw_fpeak"] = float(fG[np.argmax(h2O)])
    for ng in ("G", "NG"):
        lM, psi, ft, info = up.mass_function(lnk, P, ng)
        rec[f"f_{ng}"] = float(ft)
        if ft > 0:
            ev = lM + MSUN_LG < np.log10(5.1e14)
            psi_now = np.where(ev, 0.0, psi)
            fnow = float(np.sum(psi_now) * (lM[1] - lM[0]) * np.log(10))
            cur = emf.integrals(lM, psi_now, emf.CURRENT); fut = emf.integrals(lM, psi_now, emf.FUTURE)
            okev = bool(evap.ok(lM + MSUN_LG, np.where(ev, psi, 0.0))[0])
            rec[f"ok_{ng}"] = bool(max(cur.values()) <= 1) and okev
            rec[f"det_{ng}"] = bool(max(fut.values()) > 1)
            rec[f"fnow_{ng}"] = fnow
            rec[f"lMpk_{ng}"] = float(lM[np.argmax(psi_now)]) if fnow > 0 else float(lM[np.argmax(psi)])
            rec[f"psi_{ng}"] = psi.astype(np.float32)
        else:
            rec.update({f"ok_{ng}": True, f"det_{ng}": False, f"fnow_{ng}": 0.0, f"lMpk_{ng}": np.nan,
                        f"psi_{ng}": np.zeros(len(lM), np.float32)})
        rec[f"Cth"] = info["Cth"]; rec["alpha_shape"] = info["alpha"]
    return rec

def node(args):
    phi0, s = args
    t0 = time.time()
    Ac = critical_A(phi0, s)
    # locate P_max(delta) from two probes, P ~ delta^-p
    probes = {}
    for d in (1e-2, 1e-3):
        try:
            probes[d] = evaluate(phi0, s, Ac * (1 - d))
        except Exception:
            probes[d] = None
    pts = [(d, r) for d, r in probes.items() if r is not None]
    if len(pts) == 2 and pts[0][1]["Pmax"] > 0 and pts[1][1]["Pmax"] > pts[0][1]["Pmax"]:
        p = np.log(pts[1][1]["Pmax"] / pts[0][1]["Pmax"]) / np.log(10)
        lP0 = np.log10(pts[0][1]["Pmax"])
        # delta for target P: log P = lP0 + p (log 1e-2 - log d)
        dl = lambda Pt: 10**(-2 - (np.log10(Pt) - lP0) / p)
        dgrid = np.logspace(np.log10(dl(1e-4)), np.log10(dl(0.3)), 24)
    else:
        dgrid = np.logspace(-2, -7, 18)
    recs = []
    for d in list(dgrid):
        if d <= 0 or d >= 0.5:
            continue
        try:
            r = evaluate(phi0, s, Ac * (1 - d))
        except Exception as e:
            r = None
        if r is not None:
            r["delta"] = d
            recs.append(r)
    for d, r in probes.items():
        if r is not None:
            r["delta"] = d; recs.append(r)
    recs.sort(key=lambda r: -r["delta"])
    return dict(phi0=phi0, s=s, Ac=Ac, recs=recs, time=time.time() - t0)

if __name__ == "__main__":
    from multiprocessing import Pool
    import os
    # PBH_USR_DPHI: spacing of the bump position (default 0.2 -> 11 positions at 3.0..5.0; finer grids use cell
    # centres covering the box [2.9, 5.1] exactly); PBH_USR_NODES: output file
    DPH = float(os.environ.get("PBH_USR_DPHI", "0.2")); NODEF = os.environ.get("PBH_USR_NODES", "usr_nodes.npy")
    phis = np.round(np.arange(3.0, 5.01, 0.2), 2) if DPH == 0.2 else np.round(np.arange(2.9 + DPH / 2, 5.1, DPH), 4)
    lss = np.array([-2.2, -2.0, -1.8, -1.6, -1.4])
    jobs = [(p, 10**l) for p in phis for l in lss]
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        jobs = [(4.0, 0.02)]
    out = []
    import os
    if os.path.exists(NODEF) and not (len(sys.argv) > 1 and sys.argv[1] == "test"):
        out = list(np.load(NODEF, allow_pickle=True))
        done = {(round(n["phi0"], 4), round(n["s"], 6)) for n in out}
        jobs = [j for j in jobs if (round(j[0], 4), round(j[1], 6)) not in done]
        print(f"resuming: {len(out)} done, {len(jobs)} to go", flush=True)
    with Pool(NPROC) as pool:
        for res in pool.imap_unordered(node, jobs):
            out.append(res)
            ok = [r for r in res["recs"] if r["Pmax"] > 1e-3]
            print(f"phi0={res['phi0']:.2f} s={res['s']:.4f} A_c={res['Ac']:.4e} n={len(res['recs'])} "
                  f"t={res['time']:.0f}s  " + (f"lkp={ok[0]['lkp']:.1f} ns={ok[0]['ns']:.4f}" if ok else ""), flush=True)
            np.save(NODEF, np.array(out, dtype=object), allow_pickle=True)
