r"""
PBHs from a supercooled first-order phase transition (FOPT) and from biased domain-wall (DW) networks,
on the fixed Starobinsky background (T_reh = 2.5e9 GeV: all transitions / network formation after reheating).

=============================== ASSUMPTIONS ===============================
Common
  * Radiation domination around the event; g_rho, g_s from Saikawa-Shirai fits (gstar_ss.py); h = 0.674.
  * Non-PBH observational constraints (define the allowed region and hence the Gaussian prior):
      - LVK O3 isotropic SGWB: Omega_GW(f) < 5.8e-9 for 20 <= f <= 76.6 Hz (flat-spectrum 95% limit applied
        pointwise -- conservative simplification; LVK arXiv:2101.12130).
      - PTA: model may not exceed the NANOGrav 15-yr signal, Omega_GW(f) < 1.51e-8 (f/f_yr)^{2/3} for
        2e-9 <= f <= 3e-8 Hz (upper edge 9.3+5.8 e-9; arXiv:2306.16213). Treated as an upper limit only.
      - BBN: T_eq (FOPT) or T_ann (DW) > 5 MeV.
      - Starobinsky: T_eq < T_reh ; DW formation temperature sigma^{1/3} < T_reh.
FOPT (Gouttenoire & Volansky, arXiv:2305.04942)
  * Strong supercooling, alpha >~ 1e2, v_w = 1, instant reheating to T_eq.
  * P_coll = exp[-a (beta/H)^b (1+delta_c)^{c beta/H}], a=1.024, b=0.6921, c=0.8831 (their Eq. 15);
    f_PBH = (P_coll / 2.2e-8)(T_eq / 140 MeV) (Eq. 16); M = M_sun (20/g_*)^{1/2}(140 MeV/T_eq)^2 (Eq. 17).
    Fit validated for beta/H in [3, 8]; extrapolated beyond (P_coll negligible there).
    delta_c band: 0.40 / 0.50 (central) / 0.667.
  * GW: envelope approximation (Caprini et al. arXiv:1512.06239), kappa = 1, alpha/(1+alpha) = 1:
    h^2 Omega = 1.67e-5 (H/beta)^2 (100/g)^{1/3} [0.11/(1.42)] S_env, S_env = 3.8 x^{2.8}/(1 + 2.8 x^{3.8}),
    f_env = 16.5e-6 Hz (0.62/2.7)(beta/H)(T_eq/100 GeV)(g/100)^{1/6}.
  * Covariant treatment (Franciolini, Gouttenoire & Jinno, arXiv:2503.01962, PRL 136 (2026) 171404):
    comoving-gauge density contrast with P(delta) = P0 exp[(eps/2)(delta-mu) - 2/(eps^2 sig^2) (1 - e^{(eps/2)(delta-mu)})^2]
    (their Eq. 24), Table I at k = 0.9 k_max for beta/H = 1, 2, 4, 8 (interpolated in ln beta/H; held at the
    beta/H = 8 values above 8, which over-estimates P). Threshold delta_c^(C) in [0.40, 0.67];
    beta = P0 * int_{delta_c} P dd (Laplace estimate of the tail);  f ~ (beta/1e-10)(T/GeV) (their Eq. 22);
    M_k = 4.8e-2 M_sun (106.75/g)^{1/2} (GeV/T)^2 (their Eq. 23).
DW (Gouttenoire & Vitagliano, arXiv:2306.17841)
  * Parameters T_ann, alpha_ann (DW energy fraction at annihilation). C_d = 3, A = 0.8, p = 0.5.
  * t_dom = t_ann / sqrt(C_d alpha_ann);  R/t_ann = 0.780 L^2 - 0.618 L + 0.407, L = log10 alpha_ann (Eq. 10);
    r = (R/t_ann)/(L_corr/t);  s_ball/r^3 = 8 + [(1+tanh(a1 ln(r/a2)))/2](4 pi/3 - 8), a1 = 1.15, a2 = 5.55
    (Eq. 12; GV write "log" without a base [2311.07670 Eq. 32]; natural log verified by direct lattice-cell
    counting, which it matches far better than log10 -- see check_sball_base()); F = s p^s (Eq. 11);
    f_PBH = (T_dom / 1 TeV) F / 8e-13 (Eq. 13);  M = t_dom/(2G) = 4 pi M_P^2 t_dom (Eq. 14).
    L_corr/t band: 0.5 / 0.8 (central) / 1.2.  (Ferreira et al. arXiv:2401.14331: very large uncertainties.)
  * sigma = 3 alpha_ann M_P^2 / (4 A t_ann); formation T_form ~ sigma^{1/3}.
  * GW (GV supplement App. B.1): h^2 Omega = D (3/32 pi) eps alpha_ann^2 S(f), eps = 0.7,
    D = 3.49e-5 (g_*/10.75)(10.75/g_s)^{4/3},  S = 2/[(f/f_p) + (f_p/f)^3],
    f_p = 1.08 nHz (g_*/10.75)^{1/2}(10.75/g_s)^{1/3}(T_ann/10 MeV)   (g_s = g_* assumed).
  * Two PTA cases:  'ul'  -- NANOGrav 15-yr used only as an upper limit (as for all other models);
                    'pta' -- DW explains the PTA signal: prior restricted to Omega_DW(f_yr) in [5.3e-9, 1.51e-8]
                             (NANOGrav 9.3 +5.8/-4.0 e-9), PTA upper limit dropped, LVK/BBN/formation kept.
===========================================================================
"""
from paths import PBH_ROOT, PBHBOUNDS, CEFFYL, NG15, NPROC
import numpy as np
import curvaton_pipeline as cp

MP, h = cp.MPl, 0.674
Msun_g = 1.98847e33
GeV_to_g = 1.78266e-24
GeVinv_to_s = 6.582e-25
TREH = cp.BG["Treh"]
rng = np.random.default_rng(42)

# ---------------- GW constraints ----------------
f_LVK = np.logspace(np.log10(20), np.log10(76.6), 25)
f_PTA = np.logspace(np.log10(2e-9), np.log10(3e-8), 25)
fyr = 1 / 3.15576e7
PTA_MODE = "L"     # "L": NG15 free-spectrum likelihood (2026-11-14, default); "pt": old pointwise upper limit

def pta_ok_L(Omega_fn, dlnL=1.92, frac_skip=1e-2):
    """PTA as an upper limit in the presence of the signal (NANOGrav 15-yr HD free spectrum, 14 bins):
    SMBHB (gamma = 13/3, amplitude profiled) + X must not fit worse than SMBHB alone by more than dlnL
    (1.92 = 95%, 1 d.o.f.).  Samples whose h^2 Omega_X is < frac_skip of the best-fit SMBHB spectrum in every
    bin are passed without evaluation (their dlnL is negligible)."""
    import pta_like as pl
    if not hasattr(pta_ok_L, "_F"):
        F = pl.FreeSpec("NG15")
        lA = np.linspace(-17, -13.5, 141)
        Lpl = F.lnL(pl.h2Omega_powerlaw(F.f, lA))
        pta_ok_L._F = F
        pta_ok_L._LB = Lpl.max()
        pta_ok_L._ref = pl.h2Omega_powerlaw(F.f, lA[Lpl.argmax()])
    F = pta_ok_L._F
    h2O = Omega_fn(F.f) * h**2
    ok = np.ones(len(h2O), bool)
    rel = np.any(h2O > frac_skip * pta_ok_L._ref[None, :], axis=1)
    if rel.any():
        ok[rel] = pl.profile_smbhb(F, h2O[rel]) >= pta_ok_L._LB - dlnL
    return ok

def gw_ok(Omega_fn):
    """Omega_fn(f) -> Omega_GW (not h^2), vectorised over samples: returns bool per sample.
    LVK O3: pointwise 95% bound.  PTA: NG15 likelihood (PTA_MODE='L') or pointwise bound (PTA_MODE='pt')."""
    lvk = np.all(Omega_fn(f_LVK) < 5.8e-9, axis=-1)
    if PTA_MODE == "L":
        pta = pta_ok_L(Omega_fn)
    else:
        pta = np.all(Omega_fn(f_PTA) < 1.51e-8 * (f_PTA / fyr)**(2 / 3), axis=-1)
    return lvk & pta

# ---------------- FOPT ----------------
FOPT_BOX = np.array([[np.log10(5e-3), np.log10(TREH)],   # log10 T_eq / GeV
                     [3.0, 30.0]])                         # beta / H
# Alternative GW spectra for strongly supercooled transitions (referee response, 2026-11-16): environment variable
# PBH_FOPT_GW = "env" (default, envelope approximation, Caprini et al. 1512.06239) | "LV3" | "LV2":
# Lewicki & Vaskonen, arXiv:2208.11697 (EPJC 83 (2023) 109), bubble collisions with T_rr ~ R^-3 (gauged symmetry)
# or R^-2 (global symmetry): Omega = (H/beta)^2 [kappa alpha/(1+alpha)]^2 S,  S = A (a+b)^c / [b x^(-a/c) + a x^(b/c)]^c,
# x = f/f_p (their Eqs. 18, 24, Table 1), kappa alpha/(1+alpha) = 1 as for the envelope case.  Today:
# h^2 Omega_0 = 1.67e-5 (100/g)^{1/3} Omega,  f_p,0 = h_* (f_p/beta)(beta/H),  h_* = 1.65e-5 Hz (T/100 GeV)(g/100)^{1/6};
# for f < h_* (scales above the horizon at the transition) the spectrum is continued as f^3 (causality).
import os as _os
FOPT_GW = _os.environ.get("PBH_FOPT_GW", "env")
_LV = {"LV3": dict(A=5.13e-2, a=2.41, b=2.42, c=4.08, w=0.77), "LV2": dict(A=5.93e-2, a=1.03, b=1.84, c=1.45, w=0.64)}
def _fopt_gw_lv(lT, bH, p):
    T = 10**lT; gs = cp.gstar_s(T)
    hst = 1.65e-5 * (T / 100) * (gs / 100)**(1 / 6)
    fp = hst * p["w"] / (2 * np.pi) * bH
    amp = 1.67e-5 * (1 / bH)**2 * (100 / gs)**(1 / 3) / h**2 * p["A"] * (p["a"] + p["b"])**p["c"]
    S = lambda x: (p["b"] * x**(-p["a"] / p["c"]) + p["a"] * x**(p["b"] / p["c"]))**(-p["c"])
    def Om(f):
        x = f[None, :] / fp[:, None]; xh = (hst / fp)[:, None]
        return amp[:, None] * np.where(x >= xh, S(x), S(xh) * (x / xh)**3)
    return Om
def fopt_gw(lT, bH):
    if FOPT_GW in _LV:
        return _fopt_gw_lv(lT, bH, _LV[FOPT_GW])
    T = 10**lT; g = cp.gstar(T); gs = cp.gstar_s(T)
    fenv = 16.5e-6 * (0.62 / 2.7) * bH * (T / 100) * (gs / 100)**(1 / 6)
    amp = 1.67e-5 * (1 / bH)**2 * (100 / gs)**(1 / 3) * (0.11 / 1.42) / h**2
    def Om(f):
        x = f[None, :] / fenv[:, None]
        return amp[:, None] * 3.8 * x**2.8 / (1 + 2.8 * x**3.8)
    return Om
def fopt_allowed(P):
    lT, bH = P.T
    return (10**lT > 5e-3) & (10**lT < TREH) & (bH >= 3) & gw_ok(fopt_gw(lT, bH))
def fopt_map(P, delta_c=0.5):
    lT, bH = P.T
    T = 10**lT
    lnP = -1.024 * bH**0.6921 * (1 + delta_c)**(0.8831 * bH)
    logf = (lnP / np.log(10)) - np.log10(2.2e-8) + np.log10(T / 0.140)
    logM = 0.5 * np.log10(20 / cp.gstar(T)) + 2 * np.log10(0.140 / T)
    return logM, logf

# FGJ Table I (k = 0.9 k_max)
_FGJ = np.array([[1, 0.99, -0.35, 9.1, 1.0], [2, 0.66, -0.14, 21, 0.52],
                 [4, 0.084, -0.0042, 48, 0.041], [8, 0.024, 4.9e-4, 1.1e2, 0.0091]])
def _fgj_params(bH):
    x = np.log(np.clip(bH, 1, 8)); xs = np.log(_FGJ[:, 0])
    P0 = np.exp(np.interp(x, xs, np.log(_FGJ[:, 1])))
    mu = np.interp(x, xs, _FGJ[:, 2])
    eps = np.exp(np.interp(x, xs, np.log(_FGJ[:, 3])))
    sig = np.exp(np.interp(x, xs, np.log(_FGJ[:, 4])))
    return P0, mu, eps, sig
def _lnP(d, mu, eps, sig):
    u = 0.5 * eps * (d - mu)
    return u - 2 / (eps**2 * sig**2) * (1 - np.exp(np.minimum(u, 700)))**2
def fopt_map_cov(P, delta_c=0.5):
    lT, bH = P.T
    T = 10**lT
    P0, mu, eps, sig = _fgj_params(bH)
    # normalisation of the fit on a grid; tail by Laplace (integrand decreases monotonically beyond the mode)
    btab = np.linspace(1, 8, 141)
    dg = np.linspace(-3, 3, 6001)
    tab = []
    for b in btab:
        _, m, e, s_ = _fgj_params(np.array([b]))
        lp = _lnP(dg, m[0], e[0], s_[0]); mx = lp.max()
        tab.append(np.log(np.trapezoid(np.exp(lp - mx), dg)) + mx)
    lnNorm = np.interp(np.clip(bH, 1, 8), btab, np.array(tab))
    lp = _lnP(delta_c, mu, eps, sig)
    u = 0.5 * eps * (delta_c - mu)
    slope = 0.5 * eps * (1 + 4 / (eps**2 * sig**2) * (1 - np.exp(np.minimum(u, 700))) * np.exp(np.minimum(u, 700)))
    ln_tail = lp - np.log(np.abs(slope)) - lnNorm
    lnbeta = np.log(P0) + ln_tail
    logf = lnbeta / np.log(10) + 10 + np.log10(T)
    logM = np.log10(4.8e-2 * (106.75 / cp.gstar(T))**0.5 / T**2)
    return logM, logf

def fopt_tail_scale(bH, delta_c):
    """s = -1/(d ln P_coll / d delta_c) from the GV fit (Eq. 15)."""
    dlnP = -1.024 * bH**0.6921 * 0.8831 * bH * (1 + delta_c)**(0.8831 * bH - 1)
    return -1 / dlnP

def fopt_flags(P, delta_c):
    """extended (critical-collapse) treatment for GV FOPT samples: logM_star, logf, ok_now, det_future."""
    import emf
    lT, bH = P.T
    T = 10**lT
    _, logf = fopt_map(P, delta_c)
    logMH = np.log10(4.8e-2 * (106.75 / cp.gstar(T))**0.5 / T**2)
    s_ = fopt_tail_scale(bH, delta_c)
    n = len(P)
    logMs = np.log10(emf.KCC) + logMH + emf.GAMMA * np.log10((1 + emf.GAMMA) * s_)
    ok = np.ones(n, bool); det = np.zeros(n, bool)
    sel = logf > -12
    if sel.any():
        import evap
        lgM, psi, _ = emf.fopt_emf(logf[sel], logMH[sel], s_[sel])
        lg_g = lgM + np.log10(1.98847e33)
        ev = lg_g < np.log10(5.1e14)                       # already evaporated part of the MF
        Ic = emf.integrals(lgM, np.where(ev, 0.0, psi), emf.CURRENT)
        If = emf.integrals(lgM, np.where(ev, 0.0, psi), emf.FUTURE)
        ok_ev = evap.ok(lg_g, np.where(ev, psi, 0.0))[0]   # BBN/CMB/EGB limits on evaporated PBHs (evap.py)
        ok[sel] = np.all(np.array([Ic[k] <= 1 for k in Ic]), axis=0) & ok_ev
        det[sel] = np.any(np.array([If[k] > 1 for k in If]), axis=0)
    return logMs, logf, ok, det

def dw_flags(P, Lt):
    """DW: monochromatic at t_dom/(2G)."""
    import emf
    logM, logf = dw_map(P, Lt=Lt)
    cur = emf.mono_flags(logM, logf, emf.CURRENT); fut = emf.mono_flags(logM, logf, emf.FUTURE)
    ok = ~np.any(np.array(list(cur.values())), axis=0)
    det = np.any(np.array(list(fut.values())), axis=0)
    return logM, logf, ok, det

def dw_flags_cc(P, Lt):
    """DW with a finite mass-function width (2026-11-14).
    GV (2311.07670) expect a nearly monochromatic MF and leave the width to future work.  Width model (ours, 要確認):
      late-annihilators with r = R/L slightly above the threshold r_PBH collapse with critical scaling
      M = K M_GV Delta^gamma (K = 3, gamma = 0.36), Delta = (r - r_PBH)/r_PBH, and the percolation weight
      p^{s_ball(r)} makes Delta exponentially distributed with scale s_D = 1/(|ln p| dS/dln r) at r_PBH
      (dS/dln r = numerical derivative of s_ball).  Total f kept equal to GV's.  Same machinery as FOPT (emf.fopt_emf)."""
    import emf, evap
    logM, logf = dw_map(P, Lt=Lt)
    Tann, a, tann, tdom, sigma = dw_quantities(P)
    L = np.log10(a)
    r = (0.780 * L**2 - 0.618 * L + 0.407) / Lt
    dS = (s_ball(r * 1.001) - s_ball(r / 1.001)) / (2 * np.log(1.001))
    sD = 1 / (abs(np.log(pocc)) * dS)
    n = len(P); ok = np.ones(n, bool); det = np.zeros(n, bool)
    logMs = np.log10(emf.KCC) + logM + emf.GAMMA * np.log10((1 + emf.GAMMA) * sD)
    sel = logf > -12
    if sel.any():
        lgM, psi, _ = emf.fopt_emf(logf[sel], logM[sel], sD[sel])
        lg_g = lgM + np.log10(Msun_g)
        ev = lg_g < np.log10(5.1e14)
        Ic = emf.integrals(lgM, np.where(ev, 0.0, psi), emf.CURRENT)
        If = emf.integrals(lgM, np.where(ev, 0.0, psi), emf.FUTURE)
        ok[sel] = np.all(np.array([Ic[k] <= 1 for k in Ic]), axis=0) & evap.ok(lg_g, np.where(ev, psi, 0.0))[0]
        det[sel] = np.any(np.array([If[k] > 1 for k in If]), axis=0)
    return logMs, logf, ok, det, sD

def check_sball_base():
    """count unit cells intersecting a ball of radius r centred on a lattice vertex; compare with both fits."""
    out = []
    for r in [1, 2, 3, 5, 8, 12, 20, 30, 50]:
        n = int(np.ceil(r)) + 2; g = np.arange(-n, n)
        I, J, K = np.meshgrid(g, g, g, indexing="ij")
        dist = lambda lo: np.maximum(0, np.maximum(lo, -(lo + 1)))
        cnt = np.sum(dist(I)**2 + dist(J)**2 + dist(K)**2 < r * r) / r**3
        fe = 8 + 0.5 * (1 + np.tanh(1.15 * np.log(r / 5.55))) * (4 * np.pi / 3 - 8)
        f10 = 8 + 0.5 * (1 + np.tanh(1.15 * np.log10(r / 5.55))) * (4 * np.pi / 3 - 8)
        out.append((r, cnt, fe, f10))
    return np.array(out)

# ---------------- DW ----------------
DW_BOX = np.array([[np.log10(5e-3), 9.0],     # log10 T_ann / GeV
                   [-4.0, 0.0]])              # log10 alpha_ann
Cd, Acal, pocc = 3.0, 0.8, 0.5
def t_of_T(T):
    return 1 / (2 * cp.H_rad(T))            # GeV^-1
def T_of_t(t):
    # invert H_rad(T) = 1/(2t) on a table
    Tt = np.logspace(-4, 12, 4000)
    return np.exp(np.interp(-np.log(t), np.log(2 * cp.H_rad(Tt)), np.log(Tt)))
def dw_quantities(P):
    lT, la = P.T
    Tann, a = 10**lT, 10**la
    tann = t_of_T(Tann)
    tdom = tann / np.sqrt(Cd * a)
    sigma = 3 * a * MP**2 / (4 * Acal * tann)
    return Tann, a, tann, tdom, sigma
def dw_gw(lT, la):
    Tann, a = 10**lT, 10**la
    g = cp.gstar(Tann); gs = cp.gstar_s(Tann)
    fp = 1.08e-9 * (g / 10.75)**0.5 * (10.75 / gs)**(1 / 3) * (Tann / 0.010)
    D = 3.49e-5 * (g / 10.75) * (10.75 / gs)**(4 / 3)
    amp = D * (3 / (32 * np.pi)) * 0.7 * a**2 / h**2
    def Om(f):
        x = f[None, :] / fp[:, None]
        return amp[:, None] * 2 / (x + x**-3)
    return Om
def dw_allowed(P, case="ul"):
    Tann, a, tann, tdom, sigma = dw_quantities(P)
    base = (Tann > 5e-3) & (a < 1) & (sigma**(1 / 3) < TREH)
    Om = dw_gw(*P.T)
    lvk = np.all(Om(f_LVK) < 5.8e-9, axis=-1)
    if case == "ul":
        pta = np.all(Om(f_PTA) < 1.51e-8 * (f_PTA / fyr)**(2 / 3), axis=-1)
    elif case in ("ptaL", "ptaLE"):
        # free-spectrum likelihood (pta_like.py): DW explains the signal <=> lnL_DW >= max lnL_DW - 3.0 (2 par., ~95%)
        # ptaL: NANOGrav 15-yr HD, 14 bins (main, 2026-11-14);  ptaLE: EPTA DR2new HD, 9 bins (cross-check)
        import pta_like as pl
        nm = "NG15" if case == "ptaL" else "EPTA"
        key = "_F" + nm
        if not hasattr(dw_allowed, key):
            setattr(dw_allowed, key, (pl.FreeSpec(nm), float(np.load(PBH_ROOT + f"/dw_pta_grid_{nm}.npz")["Lmax"])))
        F, Lmax = getattr(dw_allowed, key)
        pta = F.lnL(Om(F.f) * h**2) >= Lmax - 3.0
    elif case == "ulL":
        # PTA as an upper limit in the presence of the signal (NG15): SMBHB (gamma=13/3, amplitude profiled) + DW
        # must not fit worse than SMBHB alone by more than dlnL = 1.92 (95%, 1 d.o.f.).  Bilinear interpolation
        # of the grid from dw_pta_fit.py (log T_ann in [log 5e-3, 9], log alpha in [-4, 0]).
        from scipy.interpolate import RegularGridInterpolator
        if not hasattr(dw_allowed, "_ulL"):
            g = np.load(PBH_ROOT + "/dw_pta_grid_NG15.npz")
            dw_allowed._ulL = (RegularGridInterpolator((g["lT"], g["la"]), g["LDWB"], bounds_error=False,
                                                       fill_value=None), float(g["LB"]))
        itp, LB = dw_allowed._ulL
        pta = itp(np.clip(P, [np.log10(5e-3), -4], [9, 0])) >= LB - 1.92
    else:
        oy = Om(np.array([fyr]))[:, 0]
        pta = (oy > 5.3e-9) & (oy < 1.51e-8)
    return base & lvk & pta
def s_ball(r):
    a1, a2 = 1.15, 5.55
    return r**3 * (8 + 0.5 * (1 + np.tanh(a1 * np.log(r / a2))) * (4 * np.pi / 3 - 8))
def dw_map(P, Lt=0.8):
    Tann, a, tann, tdom, sigma = dw_quantities(P)
    L = np.log10(a)
    R_over_t = 0.780 * L**2 - 0.618 * L + 0.407
    r = R_over_t / Lt
    s = s_ball(r)
    log10F = np.log10(s) + s * np.log10(pocc)
    Tdom = T_of_t(tdom)
    logf = np.log10(Tdom / 1e3) + log10F - np.log10(8e-13)
    M = 4 * np.pi * MP**2 * tdom * GeV_to_g / Msun_g
    return np.log10(M), logf

# ---------------- prior: Gaussian on the allowed region ----------------
def gaussian_prior(box, allowed_fn, nbox=400_000, nsamp=1_000_000):
    P = box[:, 0] + (box[:, 1] - box[:, 0]) * rng.random((nbox, len(box)))
    A = P[allowed_fn(P)]
    mu, cov = A.mean(0), np.cov(A.T)
    out, tot = [], 0
    while tot < nsamp:
        Q = rng.multivariate_normal(mu, cov, size=2 * nsamp)
        Q = Q[np.all((Q >= box[:, 0]) & (Q <= box[:, 1]), axis=1)]
        Q = Q[allowed_fn(Q)]
        out.append(Q); tot += len(Q)
    return mu, cov, len(A) / nbox, np.concatenate(out)[:nsamp], A

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    c = check_sball_base()
    print("s_ball check: rms(ln fit) =", np.sqrt(np.mean((c[:, 2] - c[:, 1])**2)).round(3),
          " rms(log10 fit) =", np.sqrt(np.mean((c[:, 3] - c[:, 1])**2)).round(3))
    res = {}
    # FOPT: non-covariant (GV) and covariant (FGJ)
    mu, cov, acc, S, A = gaussian_prior(FOPT_BOX, fopt_allowed)
    print(f"== FOPT: allowed fraction {acc:.3f}; prior mean {np.round(mu, 3)}, std {np.round(np.sqrt(np.diag(cov)), 3)}")
    res["FOPT_S"] = S
    for tag, fn in [("GV", fopt_map), ("cov", fopt_map_cov)]:
        for dc in (0.4, 0.5, 0.667):
            lM, lf = fn(S, dc)
            print(f"   {tag:3s} delta_c={dc}: P(f>1e-12)={np.mean(lf > -12):.4f}  P(1e-12<f<1)={np.mean((lf > -12) & (lf < 0)):.4f}"
                  f"  max log10 f={np.nanmax(lf):.3g}")
            res[f"FOPT_{tag}_{dc}"] = np.vstack([lM, lf])
            if tag == "GV":
                lMs, lf2, ok, det = fopt_flags(S, dc)
                import emf
                mono_ok = ~np.any(np.array(list(emf.mono_flags(lM, lf, emf.CURRENT).values())), axis=0)
                pb = lf > -12
                print(f"        extended MF: P(f>1e-12 & allowed) = {np.mean(pb & ok):.4f}  (monochromatic: {np.mean(pb & mono_ok):.4f});"
                      f"  P(allowed & detectable) = {np.mean(pb & ok & det):.4f}")
                res[f"FOPT_GVx_{dc}"] = np.vstack([lMs, lf2, ok, det])
    # DW: two PTA cases
    for case in ("ul", "pta"):
        mu, cov, acc, S, A = gaussian_prior(DW_BOX, lambda P: dw_allowed(P, case))
        print(f"== DW[{case}]: allowed fraction {acc:.4f}; prior mean {np.round(mu, 3)}, std {np.round(np.sqrt(np.diag(cov)), 3)}")
        res[f"DW_{case}_S"] = S
        for Lt in (0.5, 0.8, 1.2):
            lM, lf = dw_map(S, Lt=Lt)
            _, _, okd, detd = dw_flags(S, Lt)
            res[f"DW_{case}_x_{Lt}"] = np.vstack([lM, lf, okd, detd])
            sel = lf > -12
            print(f"   L/t={Lt}: P(f>1e-12)={np.mean(sel):.4f}  P(1e-12<f<1)={np.mean(sel & (lf < 0)):.4f}"
                  f"  P(allowed)={np.mean(sel & okd):.4f}  P(allowed & detectable)={np.mean(sel & okd & detd):.4f}"
                  f"  logM 5-50-95% (f>1e-12): {np.round(np.percentile(lM[sel], [5, 50, 95]), 2) if sel.any() else None}")
            res[f"DW_{case}_{Lt}"] = np.vstack([lM, lf])
    np.savez(PBH_ROOT + "/fopt_dw_samples.npz", **res)
