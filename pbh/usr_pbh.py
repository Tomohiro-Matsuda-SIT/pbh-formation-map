r"""
From a curvature power spectrum P_R(k) to (i) PBH mass function (compaction-function threshold statistics),
(ii) scalar-induced GWs, (iii) mu distortion.

=============================== ASSUMPTIONS ===============================
Threshold (Musco, De Luca, Franciolini, Riotto, PRD 103 (2021) 063538, arXiv:2011.03014):
  r_m from  int dk k^2 [(k^2 r^2 - 1) sin(kr)/(kr) + cos(kr)] P(k) = 0   (Eq. 34, P(k) = 2 pi^2 P_R / k^3);
  alpha_G = -(1/4)[1 + r_m int dk k^4 cos(k r_m) P / int dk k^3 sin(k r_m) P]   (Eq. 37);
  alpha from F(alpha)[1+F(alpha)] alpha = 2 alpha_G (Eqs. 29-30);  C_th = delta_c(alpha), Eq. 19.
  The shape (alpha, k_peak r_m) is evaluated once for the spectrum around its peak and applied at all M_H.
Abundance (Ferrante, Franciolini, Iovino, Urbano, PRD 107 (2023) 043520, arXiv:2211.01728, Eqs. 44-66):
  C = C_l - (3/8) C_l^2,  C_l = C_G dF/dzeta_G,   joint Gaussian (C_G, zeta_G) with sigma_c, sigma_r, sigma_cr
  (top-hat W, W_s = sin(kr)/kr, radiation transfer T, tau = r_m), domain C > C_th and C_l < 4/3,
  M = K M_H (C - C_th)^gamma, K = 4.36 (broad spectra), gamma = 0.36;  M_H where a H r_m = 1, r_m = kappa/k.
  Non-Gaussianity bands:  'G'  : zeta = zeta_G (realistic for a SMOOTH USR -> SR transition, where f_NL relaxes
                                 to slow-roll values; Cai et al. JCAP 05 (2018) 012, arXiv:1712.09998);
                          'NG' : zeta = -(1/3) ln(1 - 3 zeta_G) (non-perturbative USR delta-N with f_NL = 5/2,
                                 i.e. a SHARP transition; exponential tail; Pi & Sasaki PRL 131 (2023) 011002,
                                 arXiv:2211.13932) -- optimistic band.
  f_PBH: f = beta x (rho_tot(T) (a/a0)^3 / rho_DM0) at the horizon-crossing temperature of M_H
         (same conversion as curvaton_pipeline; g_rho, g_s from Saikawa-Shirai).
SIGW (Kohri & Terada, PRD 97 (2018) 123532, arXiv:1804.08577, Eqs. 6, 17, 26-27; radiation domination, Gaussian):
  Omega_GW,c(k) = (1/12) int_0^inf dt int_{-1}^{1} ds [t(2+t)(s^2-1)/((1-s+t)(1+s+t))]^2 I_RD^2 P(kv) P(ku);
  today: h^2 Omega_GW0 = h^2 Omega_r0 (g_rho(T)/3.36)(3.91/g_s(T))^{4/3} Omega_GW,c, h^2 Omega_r0 = 4.18e-5;
  f = 1.5463e-15 Hz (k / Mpc^-1).   (NG corrections to SIGW neglected.)
mu distortion (Chluba, Erickcek, Ben-Dayan, ApJ 758 (2012) 76, arXiv:1203.2681, Eq. 10):
  mu = 2.2 int_{1 Mpc^-1} P_R [exp(-k/5400) - exp(-(k/31.6)^2)] dln k  <  9e-5 (COBE/FIRAS 95%).
===========================================================================
"""
import numpy as np
from scipy.special import gamma as Gfun, gammaincc
from scipy.optimize import brentq
import curvaton_pipeline as cp

K_PBH, GAM = 4.36, 0.36
PHI = 2 / 3

def _Pk(lnk, P):
    return lambda k: np.exp(np.interp(np.log(k), lnk, np.log(np.maximum(P, 1e-300)), left=-700, right=-700))

def _W(x):
    return 3 * (np.sin(x) - x * np.cos(x)) / x**3

def shape(lnk, P):
    """-> kappa = k_peak r_m, alpha, C_th."""
    Pf = _Pk(lnk, P)
    kp = np.exp(lnk[np.argmax(P)])
    # restrict to the peak region and damp sub-horizon modes with the radiation transfer function at
    # tau = r_m (Musco et al. Eqs. 31-32); iterate r_m twice.  Without this, the far UV tail (k >> k_p) makes
    # the oscillatory integrals ill-conditioned.
    k = kp * np.exp(np.linspace(-6, 4, 4000))
    dk = np.gradient(k)
    rm = 2.74 / kp
    for _ in range(3):
        Tr = _W(k * rm / np.sqrt(3))
        Pk = 2 * np.pi**2 * Pf(k) / k**3 * Tr**2
        def g(r):
            x = k * r
            return np.sum(dk * k**2 * ((x**2 - 1) * np.sin(x) / x + np.cos(x)) * Pk)
        rs = np.linspace(0.5, 8, 300) / kp
        gv = np.array([g(r) for r in rs])
        idx = np.where(np.sign(gv[:-1]) != np.sign(gv[1:]))[0]
        rm = brentq(g, rs[idx[0]], rs[idx[0] + 1]) if len(idx) else 2.74 / kp
    aG = -0.25 * (1 + rm * np.sum(dk * k**4 * np.cos(k * rm) * Pk) / np.sum(dk * k**3 * np.sin(k * rm) * Pk))
    def Ffun(a):
        b = 5 / (2 * a)
        den = Gfun(b) * (1 - gammaincc(b, 1 / a))     # Gamma(b) - Gamma(b, 1/a)
        return np.sqrt(np.maximum(1 - 0.4 * np.exp(-1 / a) * a**(1 - b) / den, 0))
    try:
        al = brentq(lambda a: Ffun(a) * (1 + Ffun(a)) * a - 2 * aG, 0.05, 60)
    except ValueError:
        al = max(aG, 0.1)
    b = 5 / (2 * al)
    Cth = 4 / 15 * np.exp(-1 / al) * al**(1 - b) / (Gfun(b) * (1 - gammaincc(b, 1 / al)))
    return kp * rm, al, float(Cth)

def _W(x):
    return 3 * (np.sin(x) - x * np.cos(x)) / x**3

def variances(lnk, P, rm):
    k = np.exp(np.linspace(lnk[0], lnk[-1], 3000))[None, :]
    Pf = _Pk(lnk, P)(k)
    r = np.atleast_1d(rm)[:, None]
    x = k * r
    T = _W(x / np.sqrt(3))
    W = _W(x); Ws = np.sin(x) / x
    dl = np.log(k[0, 1] / k[0, 0])
    sc2 = (4 * PHI**2 / 9) * np.sum(x**4 * W**2 * T**2 * Pf, axis=1) * dl
    scr2 = (2 * PHI / 3) * np.sum(x**2 * W * Ws * T**2 * Pf, axis=1) * dl
    sr2 = np.sum(Ws**2 * T**2 * Pf, axis=1) * dl
    return sc2, scr2, sr2

def mass_function(lnk, P, ng="G", nMH=60, ngrid=240):
    """-> (log10 M [Msun] grid, psi = df/dlnM, f_tot, info)."""
    kappa, al, Cth = shape(lnk, P)
    kp = np.exp(lnk[np.argmax(P)])
    kH = np.exp(np.linspace(np.log(kp) - 4.0, np.log(kp) + 2.0, nMH))      # aH = 1/r_m
    rm = 1 / kH
    sc2, scr2, sr2 = variances(lnk, P, rm)
    sc, sr = np.sqrt(sc2), np.sqrt(sr2)
    gcr = np.clip(scr2 / (sc * sr), -0.999, 0.999)
    T = cp.T_of_lnk(np.log(kH))
    MH = np.interp(np.log(T), np.log(cp._Ttab), cp._M_T) / cp.gamma
    fb = np.interp(np.log(T), np.log(cp._Ttab), cp._fb_T)
    lMb = np.linspace(-25, 12, 371)
    psi = np.zeros(len(lMb) - 1)
    for i in range(nMH):
        if sc[i] <= 0 or not np.isfinite(sc[i]):
            continue
        if ng == "G":
            # zeta_G marginalises out: C depends on C_G only
            CG = np.linspace(0, 4 / 3, ngrid)
            pdf = np.exp(-CG**2 / (2 * sc2[i])) / np.sqrt(2 * np.pi * sc2[i])
            C = CG - 3 / 8 * CG**2
            w = pdf * np.gradient(CG)
            m = C > Cth
            Mr = K_PBH * (C[m] - Cth)**GAM
            beta_i = Mr * w[m]
        else:
            zg = np.linspace(-8 * sr[i], min(8 * sr[i], 1 / 3 - 1e-6), ngrid)
            CG = np.linspace(-8 * sc[i], 8 * sc[i], ngrid)
            ZG, CGg = np.meshgrid(zg, CG, indexing="ij")
            dF = 1 / (1 - 3 * ZG)
            Cl = CGg * dF
            C = Cl - 3 / 8 * Cl**2
            norm = 1 / (2 * np.pi * sc[i] * sr[i] * np.sqrt(1 - gcr[i]**2))
            pdf = norm * np.exp(-ZG**2 / (2 * sr2[i])) * np.exp(-(CGg / sc[i] - gcr[i] * ZG / sr[i])**2 / (2 * (1 - gcr[i]**2)))
            w = pdf * (zg[1] - zg[0]) * (CG[1] - CG[0])
            m = (C > Cth) & (Cl < 2 * PHI)
            Mr = K_PBH * (C[m] - Cth)**GAM
            beta_i = Mr * w[m]
        if beta_i.size == 0:
            continue
        lM = np.log10(MH[i] * Mr)
        dlnMH = np.log(kH[1] / kH[0]) * 2           # M_H ~ k^-2
        h, _ = np.histogram(lM, bins=lMb, weights=beta_i * fb[i] * dlnMH)
        psi += h
    lMc = 0.5 * (lMb[1:] + lMb[:-1])
    psi /= (lMb[1] - lMb[0]) * np.log(10)            # per lnM
    ftot = np.sum(psi) * (lMb[1] - lMb[0]) * np.log(10)
    return lMc, psi, ftot, dict(kappa=kappa, alpha=al, Cth=Cth, sigma_c_max=float(np.nanmax(sc)))

# ---------------- SIGW ----------------
_tc = np.sqrt(3) - 1
_t = np.unique(np.concatenate([np.logspace(-6, 3, 3000), _tc + np.linspace(-1e-3, 1e-3, 201)]))
_s = np.linspace(-1, 1, 201)
_T, _S = np.meshgrid(_t, _s, indexing="ij")
_U, _V = (_T + _S + 1) / 2, (_T - _S + 1) / 2
_pre = (_T * (2 + _T) * (_S**2 - 1) / ((1 - _S + _T) * (1 + _S + _T)))**2
def _I2(u, v):
    a = u**2 + v**2 - 3
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.log(np.abs((3 - (u + v)**2) / (3 - (u - v)**2)))
    return 0.5 * (3 * a / (4 * u**3 * v**3))**2 * ((-4 * u * v + a * L)**2 + np.pi**2 * a**2 * (u + v > np.sqrt(3)))
_KER = _pre * np.nan_to_num(_I2(_U, _V))
# trapezoid weights (scale-invariant check: 0.826 vs Kohri-Terada 0.8222, Eq. 31)
_wtt = np.zeros_like(_t); _wtt[1:] += 0.5 * np.diff(_t); _wtt[:-1] += 0.5 * np.diff(_t)
_wts = np.zeros_like(_s); _wts[1:] += 0.5 * np.diff(_s); _wts[:-1] += 0.5 * np.diff(_s)
_wt = _wtt[:, None] * _wts[None, :]

def sigw(lnk, P, kgrid):
    Pf = _Pk(lnk, P)
    out = np.empty(len(kgrid))
    for i, k in enumerate(kgrid):
        out[i] = np.sum(_KER * Pf(k * _V) * Pf(k * _U) * _wt) / 12
    T = cp.T_of_lnk(np.log(kgrid))
    fac = 4.18e-5 * (cp.gstar(T) / 3.36) * (3.91 / cp.gstar_s(T))**(4 / 3)
    return 1.5463e-15 * kgrid, fac * out                      # f [Hz], h^2 Omega_GW0

def mu_distortion(lnk, P):
    k = np.exp(lnk); m = k >= 1.0
    W = 2.2 * (np.exp(-k / 5400) - np.exp(-(k / 31.6)**2))
    return float(np.trapezoid((P * W)[m], lnk[m])) if m.sum() > 1 else 0.0
