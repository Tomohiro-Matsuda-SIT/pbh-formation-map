r"""
Inflating curvaton (Kohri, Lin, Matsuda, arXiv:1211.2371; Dimopoulos et al. arXiv:1110.2951) on the fixed
Starobinsky background.

=============================== ASSUMPTIONS ===============================
Timeline: Starobinsky inflation (fixed) -> MD reheating (T_reh = 2.5e9 GeV) -> RD -> at T_2 the vacuum energy V_0
  of the curvaton sector equals the radiation density (V_0 = rho_r(T_2)) -> curvaton (hilltop) inflation for N_2
  e-folds with tachyonic mass m^2 = -3 eta_2 H_2^2 -> instant reheating back to T_R2 = T_2 -> RD.
Curvaton potential  V = V_0 - (1/2)|m_2|^2 sigma^2 near sigma = 0 (hilltop), ends at sigma_end = M_P sqrt(2/eta_2).
During primordial inflation: Hubble-induced mass c_1 H^2 > 0 -> sigma decays at rate
   lam_1 = (3/2)(1 - sqrt(1 - 4 c_1/9)) per e-fold;  x = delta sigma/sigma conserved after exit
   (KLM: T_1 ~ e^{-beta_1 Delta N_1}).  Intermediate epoch: sigma frozen (KLM: beta_int ~ 0).
Curvaton inflation: sigma grows at rate g_2 = (3/2)(sqrt(1 + 4 eta_2/3) - 1) per e-fold (exact for constant H_2).
   N_2 = ln(sigma_end/sigma_ini) / g_2  (derived, not free).
delta N (non-linear):  zeta_sigma = -(1/g_2) ln|1 + x|  ;  zeta = r zeta_sigma, r = 1/(1 + (4/3) e^{-4 N_2})
   (KLM Eq. r_sigma with eps_w ~ 1 at the end of curvaton inflation; cf. Enomoto et al. arXiv:1210.7118 for NG).
Only k <= k_2 = a_2 H_2 (super-horizon at the onset of curvaton inflation) carry zeta; smaller scales re-entered
   during RD before T_2 and decayed.  Modes exiting during curvaton inflation: P ~ (H_2/(2 pi g_2 sigma))^2,
   suppressed by (H_2/H_1)^2, neglected (checked a posteriori via constraint C3b).
Re-entry: k = a H in the final RD; M = gamma M_H(T_re), gamma = 0.2.
Background shift: the extra N_2 e-folds reduce N_* by N_2 (a_end H_end/k_* rescaled by e^{-N_2}); mu and N_* are
   re-solved from A_s for each N_2 (table).  n_s from Starobinsky slow roll at N_*.
Formation: zeta > zeta_c = 1.0 (placeholder), PS, Gaussian window, gamma = 0.2.  Hilltop sigma -> 0 crossing:
   conservative = exclude |1+x| < eps_min (sub-scale fluctuation), optimistic = include.
Parameters: c_1, log10 sigma_*/H_*, log10 T_2/GeV, log10 eta_2, log10 sigma_end/M_P.
   sigma_end is free (higher-order terms, e.g. +lambda sigma^4, end curvaton inflation earlier); it is bounded by
   the pure-quadratic hilltop value M_P sqrt(2/eta_2) (variant QUAD fixes it there).
Non-PBH constraints:
   C1 CMB: curvaton P_zeta(k_*) < 0.01 A_s, |f_NL| < 5 (f_NL = (5/6) g_2 r^-1 w^2 from the log map);
      NOTE (2026-11-16): Enomoto-Kohri-Matsuda v3 Eq. (38) gives f_NL ~ (5/2) eta_2 at r = 1 (KLM Eq. 26 quotes an
      earlier form); the log map gives (5/6) g_2 ~ (5/6) eta_2.  Both satisfy |f_NL| < 5 for eta_2 < 3/4, so the
      choice does not change any result; stated in the paper's appendix.
      SCOPE (2026-11-16): the spectator lives on the CMB-generating reference background (H_* = 1.6e13 GeV).  KLM
      (arXiv:1211.2371) assume a separate low-scale primordial inflation (H_1 ~ 2e11 GeV) with the CMB generated
      elsewhere; that scenario is outside this pipeline (see infcurv_freeN2 check in docs/kohri_consistency_review.md).
       n_s > 0.9565 (Planck 2018 2 sigma lower edge of 0.9649 +- 0.0042).
   C2 spectator during primordial inflation: c_1 H^2 sigma_*^2/2 < 0.03 M_P^2 H^2, sigma_* < M_P.
   C3 classicality: (a) P_x(end of primordial inflation) < 1 ; (b) H_2/(2 pi) < g_2 sigma_ini.
   C4 T_2 in (5 MeV, T_reh); eta_2 < 3/4 (KLM |eta| < 3/4); N_2 >= 1.
===========================================================================
"""
from paths import PBH_ROOT, PBHBOUNDS, CEFFYL, NG15, NPROC
import numpy as np
from scipy.special import log_ndtr
from scipy.optimize import brentq
import curvaton_pipeline as cp

rng = np.random.default_rng(23)
PARAM = "N2"     # "N2": 5th parameter is N_2 ; "send": 5th parameter is log10 sigma_end/M_P
MP, As = cp.MPl, cp.As

# ---------- background table as a function of N_2 ----------
def bg_for(N2):
    Nst = 51.0 - N2
    for _ in range(60):
        y = cp.y_of_N(Nst)
        mu = np.sqrt(As * 24 * np.pi**2 * cp.epsV(y) / cp.V_over(y)) * MP
        Gam = mu**3 / (192 * np.pi * MP**2)
        rho_reh = 3 * MP**2 * Gam**2
        Treh = (rho_reh * 30 / (np.pi**2 * 106.75))**0.25
        rho_end = cp.V_over(cp.yend) * mu**2 * MP**2 * 1.5
        Hst = np.sqrt(cp.V_over(y) / 3) * mu
        Hend = np.sqrt(rho_end / 3) / MP
        aHend = cp.aH_Mpc(Treh) * (rho_end / rho_reh)**(1 / 6) * np.exp(-N2)
        Nnew = np.log(aHend / cp.kstar) + np.log(Hst / Hend)
        if abs(Nnew - Nst) < 1e-8:
            break
        Nst = Nnew
    eps = cp.epsV(y)
    # eta_V for Starobinsky: V''/V
    e = np.exp(-y)
    etaV = (4 / 3) * (2 * e**2 - e) / (1 - e)**2
    ns = 1 - 6 * eps + 2 * etaV
    return dict(Nst=Nst, mu=mu, Treh=Treh, Hst=Hst, Hend=Hend, lnaHend=np.log(aHend), ns=ns)

N2tab = np.linspace(0, 30, 121)
_BGt = [bg_for(n) for n in N2tab]
def bgi(key, N2):
    return np.interp(N2, N2tab, [b[key] for b in _BGt])

# universal h(N) = H/mu on a fine grid
Nfine = np.linspace(0, 55, 2201)
hfine = np.array([np.sqrt(cp.V_over(cp.y_of_N(N)) / 3) for N in Nfine])

def solve_N2(c1, lsig, eta2, lsend=None, it=40):
    """N_2 from sigma_ini(N_*(N_2)); fixed-point iteration (vectorised)."""
    lam1 = 1.5 * (1 - np.sqrt(1 - 4 * c1 / 9))
    g2 = 1.5 * (np.sqrt(1 + 4 * eta2 / 3) - 1)
    lnsend = np.log(MP * np.sqrt(2 / eta2)) if lsend is None else np.log(10**lsend * MP)
    N2 = np.full_like(c1, 5.0)
    for _ in range(it):
        Nst = bgi("Nst", N2); Hst = bgi("Hst", N2)
        lnsini = np.log(10**lsig * Hst) - lam1 * Nst
        N2new = np.clip((lnsend - lnsini) / g2, 0, 30)
        N2 = 0.5 * N2 + 0.5 * N2new
    return N2, lam1, g2, lnsini

def derived(P):
    c1, lsig, lT2, leta, p5 = P.T
    eta2 = 10**leta
    if PARAM == "N2":
        N2 = p5
        lam1 = 1.5 * (1 - np.sqrt(1 - 4 * c1 / 9))
        g2 = 1.5 * (np.sqrt(1 + 4 * eta2 / 3) - 1)
        lnsini = np.log(10**lsig * bgi("Hst", N2)) - lam1 * bgi("Nst", N2)
        lsend = (lnsini + g2 * N2 - np.log(MP)) / np.log(10)
    else:
        lsend = p5
        N2, lam1, g2, lnsini = solve_N2(c1, lsig, eta2, lsend)
    Nst, mu, Hst, Hend = bgi("Nst", N2), bgi("mu", N2), bgi("Hst", N2), bgi("Hend", N2)
    # N grid per sample: 0..N_*, use 400 points
    u = np.linspace(0, 1, 400)
    Ngrid = u[None, :] * Nst[:, None]
    H = mu[:, None] * np.interp(Ngrid, Nfine, hfine)
    lnk = bgi("lnaHend", N2)[:, None] - Ngrid + np.log(H / Hend[:, None])
    lnsig = (np.log(10**lsig * Hst))[:, None] - lam1[:, None] * (Nst[:, None] - Ngrid)
    Px = (H / (2 * np.pi))**2 * np.exp(-2 * lnsig)
    T2 = 10**lT2
    H2 = np.sqrt(np.pi**2 / 30 * cp.gstar(T2)) * T2**2 / MP           # V_0 = rho_r(T_2), H_2^2 = V_0/(3 M_P^2)
    lnk2 = np.log(cp.aH_Mpc(T2) * np.sqrt(2)) - N2                     # k_2 = a_2 H_2 (H at onset = sqrt(2) H_2)
    r = 1 / (1 + (4 / 3) * np.exp(-4 * N2))
    return dict(lsend=lsend, c1=c1, eta2=eta2, N2=N2, lam1=lam1, g2=g2, lnsini=lnsini, Nst=Nst, ns=bgi("ns", N2),
                Ngrid=Ngrid, lnk=lnk, Px=Px, T2=T2, H2=H2, lnk2=lnk2, r=r, Hst=Hst, lsig=lsig)

BOX = np.array([[0.05, 2.2],           # c_1
                [3.0, 5.2],            # log10 sigma_*/H_*  (upper edge ~ M_P/H_*)
                [np.log10(5e-3), np.log10(2.5e9)],   # log10 T_2
                [-4.0, np.log10(0.75)],              # log10 eta_2
                [1.0, 8.0]])                         # N_2 (PARAM="N2")

def constraints(P, chunk=20000):
    if len(P) > chunk:
        return np.concatenate([constraints(P[i:i + chunk]) for i in range(0, len(P), chunk)])
    d = derived(P)
    lin = (d["r"] / d["g2"])**2                       # zeta = (r/g_2) x at linear order
    Pz_cmb = lin * d["Px"][:, -1]
    w = Pz_cmb / (As + Pz_cmb)
    fNL = 5 / 6 * d["g2"] / d["r"] * w**2
    C1 = (Pz_cmb < 0.01 * As) & (np.abs(fNL) < 5) & (d["ns"] > 0.9565)
    sst = 10**d["lsig"] * d["Hst"]
    C2 = (0.5 * d["c1"] * sst**2 < 0.03 * MP**2) & (sst < MP)
    C3 = (d["Px"][:, 0] < 1) & (d["H2"] / (2 * np.pi) < d["g2"] * np.exp(d["lnsini"]))
    C4 = (d["T2"] > 5e-3) & (d["T2"] < bgi("Treh", d["N2"])) & (d["eta2"] < 0.75) & (d["N2"] >= 1) & (d["N2"] < 29.9) \
         & (10**d["lsend"] <= np.sqrt(2 / d["eta2"]))
    return C1 & C2 & C3 & C4

def _band(lo, hi, s):
    s = np.maximum(s, 1e-300)
    lo = np.broadcast_to(lo, np.shape(s)); hi = np.broadcast_to(hi, np.shape(s))
    w = hi - lo
    narrow = w < 1e-6 * s
    mid = 0.5 * (lo + hi)
    ln_n = np.where(w > 0, np.log(np.maximum(w, 1e-300)) - 0.5 * (mid / s)**2 - 0.5 * np.log(2 * np.pi) - np.log(s), -np.inf)
    a, b = lo / s, hi / s
    la = np.where(a > 0, log_ndtr(-a), log_ndtr(b))
    lb = np.where(a > 0, log_ndtr(-b), log_ndtr(a))
    ln_w = la + np.log1p(-np.exp(np.minimum(lb - la, -1e-16)))
    return np.where(narrow, ln_n, ln_w)

def pbh_map(P, zeta_c=1.0, nk=200, skip=-60, mode="con", chunk=5000):
    if len(P) > chunk:
        parts = [pbh_map(P[i:i + chunk], zeta_c, nk, skip, mode) for i in range(0, len(P), chunk)]
        return tuple(np.concatenate(x) for x in zip(*parts))
    d = derived(P)
    n = len(P)
    logM = np.full(n, np.nan); logf = np.full(n, -np.inf)
    # zeta > zeta_c  <=>  |1 + x| < q_c ,  q_c = exp(-g_2 zeta_c / r)
    qc = np.exp(-d["g2"] * zeta_c / d["r"])
    for i in range(n):
        lq_all, Pall = d["lnk"][i, ::-1], d["Px"][i, ::-1]   # ascending k
        sel = lq_all <= d["lnk2"][i]
        if sel.sum() < 3:
            continue
        lq, Pq = lq_all[sel], Pall[sel]
        s2max = np.trapezoid(Pq, lq)
        if log_ndtr(-(1 - qc[i]) / np.sqrt(s2max)) / np.log(10) < skip:
            continue
        lk = np.linspace(d["lnk2"][i] - 30, d["lnk2"][i], nk)
        W = np.exp(-np.exp(2 * (lq[None, :] - lk[:, None])))
        s = np.sqrt(np.trapezoid(Pq[None, :] * W, lq, axis=1))
        cum = np.concatenate([[0], np.cumsum(0.5 * (Pall[1:] + Pall[:-1]) * np.diff(lq_all))])
        epsmin = np.sqrt(np.maximum(cum[-1] - np.interp(lk, lq_all, cum), 1e-300))
        q = qc[i]
        # x in (-1 - q, -1 + q)
        lnb_opt = _band(-1 - q, -1 + q, s)
        okc = q > epsmin
        lnb_con = np.where(okc, np.logaddexp(_band(-1 - q, -1 - epsmin, s), _band(-1 + epsmin, -1 + q, s)), -np.inf)
        lnb = lnb_opt if mode == "opt" else lnb_con
        lnb = np.where(np.isnan(lnb), -np.inf, lnb)
        T = cp.T_of_lnk(lk)
        lnM = np.log(np.interp(np.log(T), np.log(cp._Ttab), cp._M_T))
        lndf = lnb + np.log(np.interp(np.log(T), np.log(cp._Ttab), cp._fb_T))
        if not np.isfinite(lndf).any():
            continue
        j = np.argmax(lndf)
        o = np.argsort(lnM)
        logf[i] = (np.log(np.trapezoid(np.exp(lndf[o] - lndf[j]), lnM[o])) + lndf[j]) / np.log(10)
        logM[i] = lnM[j] / np.log(10)
    return logM, logf

def build_prior(nbox=400_000):
    P = BOX[:, 0] + (BOX[:, 1] - BOX[:, 0]) * rng.random((nbox, 5))
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
    import warnings; warnings.filterwarnings("ignore")
    print("N2 -> N_*, n_s:", [(n, round(bgi('Nst', n), 2), round(bgi('ns', n), 4)) for n in [0, 2, 4, 6, 8]])
    mean, cov, acc, A = build_prior()
    print("allowed fraction:", acc, " n =", len(A))
    for nm, m_, s_ in zip(["c1", "log sig*/H*", "log T2", "log eta2", "N2"], mean, np.sqrt(np.diag(cov))):
        print(f"  {nm:12s} mean {m_:7.3f} std {s_:6.3f}")
    D = derived(A); print("  N2 range in allowed:", np.percentile(D["N2"], [0, 50, 100]).round(2))
    np.savez(PBH_ROOT + "/infcurv_prior.npz", mean=mean, cov=cov)
    out = {}
    for mode in ("con", "opt"):
        lM, lf = pbh_map(A, skip=-1e9, mode=mode)
        fin = np.isfinite(lf)
        print(f"[{mode}] uniform scan: finite {fin.sum()}, max log10 f = {lf[fin].max() if fin.any() else None},"
              f" f>1e-12: {(lf > -12).sum()}, f>1e-3: {(lf > -3).sum()}")
        out[f"lM_{mode}"], out[f"lf_{mode}"] = lM, lf
    np.savez(PBH_ROOT + "/infcurv_scan.npz", P=A, **out)
