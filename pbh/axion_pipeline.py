r"""
Axion-like curvaton (Kawasaki-Kitajima-Yanagida type, arXiv:1207.2550) on the fixed Starobinsky background.

=============================== ASSUMPTIONS ===============================
Background: identical to curvaton_pipeline.py (Starobinsky, T_reh = 2.5e9 GeV, N_* = 51).
Radial field: Hubble-induced mass c H^2 during inflation -> phi(N) = f exp[lam_phi (N - N_f)] for N > N_f,
   phi = f for N <= N_f,  lam_phi = (3/2)(1 - sqrt(1 - 4c/9))   (KKY Eq. for the blue tilt).
Phase perturbation at exit: P_dtheta(N) = (H(N) / (2 pi phi(N)))^2 ; dtheta conserved afterwards.
Curvaton potential: V = m^2 f^2 (1 - cos theta), FULL anharmonic background via G(theta) (axion_G.py):
   oscillation starts in MD reheating if m > Gamma_I (= 9.1 GeV), else in RD.
   MD: Omega(T) = (f/M_P)^2 G_MD(theta) (g_s,reh/g_s)^(1/3) T_reh/T ;  RD: Omega(T) = (f/M_P)^2 G_RD(theta) sqrt(m / 2H(T)).
   p = Omega (ratio to radiation), Omega_dec = p/(1+p).
Sudden decay, non-linear delta N:  (1-Omega_dec) e^{-4 zeta} + Omega_dec [G(theta+dtheta)/G(theta)] e^{-3 zeta} = 1.
   G is even and 2pi-periodic in theta (field crossing pi lands in the neighbouring vacuum: domain-wall issue, CAVEAT).
Only k <= k_dec contribute. Formation: zeta > zeta_c = 1.0 (placeholder), PS, Gaussian window, gamma = 0.2.
Parameters: c, N_f, log10 f/H_*, theta_i, log10 m/GeV, log10 T_dec/GeV.
Non-PBH constraints:
   C1 CMB: curvaton P_zeta(k_*) < 0.01 A_s and |f_NL| < 5 (linear: zeta = (r/3) (G'/G) dtheta, r = 3 Om/(4-Om)).
   C2 light & spectator: m < 0.1 H_end ; radial energy c H_*^2 phi_*^2 / 2 < 0.01 * 3 M_P^2 H_*^2 ; phi_* < M_P.
   C3 no hilltop crossing at 1 sigma: total dtheta variance (all modes to end of inflation) < (pi - theta_i)^2.
   C4 decay: T_dec > 5 MeV, T_dec < T_reh, H(T_dec) < m^3/(16 pi f^2) (perturbative decay, kappa <= 1, KKY).
===========================================================================
"""
from paths import PBH_ROOT, PBHBOUNDS, CEFFYL, NG15, NPROC
import numpy as np
from scipy.special import log_ndtr
import curvaton_pipeline as cp

rng = np.random.default_rng(7)
_G = np.load(PBH_ROOT + "/axion_G.npz")
_th = _G["th"]
_lGMD, _lGRD = np.log(_G["GMD"]), np.log(_G["GRD"])
# extend to theta -> 0 with harmonic scaling
_th_full = np.concatenate([[1e-8], _th])
def _ext(lG):
    return np.concatenate([[lG[0] + 2 * np.log(1e-8 / _th[0])], lG])
_lGMD_f, _lGRD_f = _ext(_lGMD), _ext(_lGRD)
# coordinate u = -ln(pi - theta) handles the hilltop
_u = -np.log(np.pi - _th_full)

# hilltop asymptotics (fit to the table for u > 15, rel. error < 0.15%):
#   MD: G^{1/2} = 1.4122 u + 2.6357 ;  RD: G^{2/3} = 2.2832 u + 3.0485 ,  u = -ln(pi - theta)
_AMD, _BMD, _ARD, _BRD = 1.4122, 2.6357, 2.2832, 3.0485
_umax = _u[-1]

def lnG_of_u(u, era_md):
    a = np.where(u <= _umax, np.interp(u, _u, _lGMD_f), 2 * np.log(_AMD * u + _BMD))
    b = np.where(u <= _umax, np.interp(u, _u, _lGRD_f), 1.5 * np.log(_ARD * u + _BRD))
    return np.where(era_md, a, b)

def lnG(theta, era_md):
    t = np.mod(np.abs(theta), 2 * np.pi)
    t = np.where(t > np.pi, 2 * np.pi - t, t)
    u = -np.log(np.maximum(np.pi - t, 1e-300))
    return lnG_of_u(u, era_md)

def u_of_lnG(lg, era_md):
    """inverse: u = -ln(pi - theta) with lnG(theta) = lg, extrapolated beyond the table."""
    a = np.where(lg <= _lGMD_f[-1], np.interp(lg, _lGMD_f, _u), (np.exp(lg / 2) - _BMD) / _AMD)
    b = np.where(lg <= _lGRD_f[-1], np.interp(lg, _lGRD_f, _u), (np.exp(2 * lg / 3) - _BRD) / _ARD)
    return np.where(era_md, a, b)

BG, Ng, Hg, lnkN = cp.BG, cp.Ngrid, cp.Hgrid, cp.lnk_of_N
MP = cp.MPl
lnG_max_MD, lnG_max_RD = _lGMD_f[-1], _lGRD_f[-1]

BOX = np.array([[0.2, 2.2],                 # c
                [0.0, BG["Nst"] - 5],       # N_f
                [-1.0, 5.0],                # log10 f/H_*
                [0.01, np.pi - 1e-3],       # theta_i
                [-20.0, np.log10(0.1 * BG["Hend"])],   # log10 m/GeV
                [np.log10(5e-3), np.log10(BG["Treh"])]])  # log10 T_dec

def derived(P):
    c, Nf, lf, thi, lm, lT = P.T
    f = 10**lf * BG["Hst"]; m = 10**lm; Tdec = 10**lT
    lam = 1.5 * (1 - np.sqrt(1 - 4 * c / 9))
    lnphi = np.log(f)[:, None] + lam[:, None] * np.maximum(Ng[None, :] - Nf[:, None], 0)
    Pdth = (Hg[None, :] / (2 * np.pi))**2 * np.exp(-2 * lnphi)
    md = m > BG["Gam"]
    lg = lnG(thi, md)
    gfac = (cp.gstar_s(BG["Treh"]) / cp.gstar_s(Tdec))**(1 / 3)
    lnOm = 2 * np.log(f / MP) + lg + np.where(md, np.log(gfac * BG["Treh"] / Tdec),
                                               0.5 * np.log(m / (2 * cp.H_rad(Tdec))))
    p = np.exp(lnOm)
    Om = p / (1 + p)
    # dlnG/dtheta by finite difference
    h = 1e-4 * np.minimum(thi, np.pi - thi)
    dlg = (lnG(thi + h, md) - lnG(thi - h, md)) / (2 * h)
    return dict(c=c, Nf=Nf, f=f, thi=thi, m=m, Tdec=Tdec, lam=lam, lnphi=lnphi, Pdth=Pdth, md=md,
                lg=lg, Om=Om, dlg=dlg)

def constraints(P, chunk=20000):
    if len(P) > chunk:
        return np.concatenate([constraints(P[i:i + chunk]) for i in range(0, len(P), chunk)])
    d = derived(P)
    r = 3 * d["Om"] / (4 - d["Om"])
    Pz_cmb = (r / 3 * d["dlg"])**2 * d["Pdth"][:, -1]
    w = Pz_cmb / (cp.As + Pz_cmb)
    fNL = 5 / (4 * np.maximum(r, 1e-300)) * w**2
    C1 = (Pz_cmb < 0.01 * cp.As) & (np.abs(fNL) < 5)
    phist = np.exp(d["lnphi"][:, -1])
    C2 = (d["m"] < 0.1 * BG["Hend"]) & (0.5 * d["c"] * phist**2 < 0.03 * MP**2) & (phist < MP)
    var_tot = np.trapezoid(d["Pdth"][:, ::-1], -Ng[::-1][None, :] if False else lnkN[::-1], axis=1)
    C3 = var_tot < (np.pi - d["thi"])**2
    C4 = (d["Tdec"] > 5e-3) & (d["Tdec"] < BG["Treh"]) & \
         (cp.H_rad(d["Tdec"]) < d["m"]**3 / (16 * np.pi * d["f"]**2))
    return C1 & C2 & C3 & C4

def pbh_map(P, zeta_c=1.0, nk=200, skip=-60, chunk=5000, mode="con"):
    if len(P) > chunk:
        parts = [pbh_map(P[i:i + chunk], zeta_c, nk, skip, mode=mode) for i in range(0, len(P), chunk)]
        return tuple(np.concatenate(x) for x in zip(*parts))
    d = derived(P)
    n = len(P)
    logM = np.full(n, np.nan); logf = np.full(n, -np.inf)
    lnk_dec = np.log(cp.aH_Mpc(d["Tdec"]))
    Om = d["Om"]
    y_c = (1 - (1 - Om) * np.exp(-4 * zeta_c)) * np.exp(3 * zeta_c) / Om
    lgc = d["lg"] + np.log(y_c)
    epsc = np.exp(-u_of_lnG(lgc, d["md"]))          # pi - theta_c  (may be astronomically small)
    lq_all = lnkN[::-1]
    for i in range(n):
        sel = lq_all <= lnk_dec[i]
        if sel.sum() < 3:
            continue
        lq, Pq = lq_all[sel], d["Pdth"][i, ::-1][sel]
        Pall = d["Pdth"][i, ::-1]
        t = d["thi"][i]
        dist = np.pi - t - epsc[i]                     # distance from theta_i to theta_c
        s2max = np.trapezoid(Pq, lq)
        if log_ndtr(-dist / np.sqrt(s2max)) / np.log(10) < skip:
            continue
        lk = np.linspace(lnk_dec[i] - 30, lnk_dec[i], nk)
        W = np.exp(-np.exp(2 * (lq[None, :] - lk[:, None])))
        s = np.sqrt(np.trapezoid(Pq[None, :] * W, lq, axis=1))
        # sub-scale fluctuation: eps_min(k)^2 = int_{q>k}^{k_end} P_dtheta dlnq
        cum = np.concatenate([[0], np.cumsum(0.5 * (Pall[1:] + Pall[:-1]) * np.diff(lq_all))])
        epsmin = np.sqrt(np.maximum(cum[-1] - np.interp(lk, lq_all, cum), 1e-300))
        e = epsc[i]
        # optimistic: theta in (pi - e, pi + e) around +pi and -pi
        lnb_opt = np.logaddexp(_band(np.pi - e - t, np.pi + e - t, s), _band(-np.pi - e - t, -np.pi + e - t, s))
        # conservative: exclude |theta -/+ pi| < eps_min
        okc = e > epsmin
        lo1, hi1 = np.pi - e - t, np.pi - epsmin - t
        lo2, hi2 = np.pi + epsmin - t, np.pi + e - t
        lnb_con = np.where(okc, np.logaddexp(np.logaddexp(_band(lo1, hi1, s), _band(lo2, hi2, s)),
                                             np.logaddexp(_band(lo1 - 2*np.pi, hi1 - 2*np.pi, s),
                                                          _band(lo2 - 2*np.pi, hi2 - 2*np.pi, s))), -np.inf)
        lnb = lnb_opt if mode == "opt" else lnb_con
        T = cp.T_of_lnk(lk)
        lnM = np.log(np.interp(np.log(T), np.log(cp._Ttab), cp._M_T))
        lndf = lnb + np.log(np.interp(np.log(T), np.log(cp._Ttab), cp._fb_T))
        if not np.isfinite(lndf).any():
            continue
        if not np.isfinite(lndf).any():
            continue
        j = np.argmax(lndf)
        o = np.argsort(lnM)
        logf[i] = (np.log(np.trapezoid(np.exp(lndf[o] - lndf[j]), lnM[o])) + lndf[j]) / np.log(10)
        logM[i] = lnM[j] / np.log(10)
    return logM, logf

def _band(lo, hi, s):
    """ln P(lo < x < hi), x ~ N(0, s^2), stable for tails and for very narrow bands."""
    s = np.maximum(s, 1e-300)
    lo, hi = np.broadcast_arrays(lo, hi)
    lo = np.broadcast_to(lo, np.shape(s)); hi = np.broadcast_to(hi, np.shape(s))
    w = hi - lo
    narrow = w < 1e-6 * s
    mid = 0.5 * (lo + hi)
    ln_narrow = np.log(np.maximum(w, 1e-300)) - 0.5 * (mid / s)**2 - 0.5 * np.log(2 * np.pi) - np.log(s)
    ln_narrow = np.where(w > 0, ln_narrow, -np.inf)
    a, b = lo / s, hi / s
    # use upper-tail form when the band is on the positive side
    la = np.where(a > 0, log_ndtr(-a), log_ndtr(b))
    lb = np.where(a > 0, log_ndtr(-b), log_ndtr(a))
    ln_wide = la + np.log1p(-np.exp(np.minimum(lb - la, -1e-16)))
    return np.where(narrow, ln_narrow, ln_wide)

def build_prior(nbox=400_000):
    P = BOX[:, 0] + (BOX[:, 1] - BOX[:, 0]) * rng.random((nbox, 6))
    ok = constraints(P)
    A = P[ok]
    return A.mean(0), np.cov(A.T), ok.mean(), A

def sample_prior(mean, cov, n):
    out, tot = [], 0
    while tot < n:
        P = rng.multivariate_normal(mean, cov, size=2 * n)
        P = P[np.all((P >= BOX[:, 0]) & (P <= BOX[:, 1]), axis=1)]
        P = P[constraints(P)]
        out.append(P); tot += len(P)
    return np.concatenate(out)[:n]

if __name__ == "__main__":
    mean, cov, acc, A = build_prior()
    print("allowed fraction:", acc)
    names = ["c", "N_f", "log f/H", "theta_i", "log m", "log Tdec"]
    for nm, mu, sd in zip(names, mean, np.sqrt(np.diag(cov))):
        print(f"  {nm:9s} mean {mu:8.3f}  std {sd:7.3f}")
    np.savez(PBH_ROOT + "/axion_prior.npz", mean=mean, cov=cov)
    # (i) the allowed region scanned uniformly: what is the best case?
    out = {}
    for mode in ("con", "opt"):
        lM, lf = pbh_map(A[:100000], skip=-1e9, mode=mode)
        fin = np.isfinite(lf)
        print(f"[{mode}] uniform scan of allowed region: finite {fin.sum()}, max log10 f = {lf[fin].max() if fin.any() else None},"
              f" f>1e-12: {(lf > -12).sum()}, f>1e-3: {(lf > -3).sum()}")
        out[f"lM_{mode}"], out[f"lf_{mode}"] = lM, lf
    np.savez(PBH_ROOT + "/axion_scan.npz", P=A[:100000], **out)
