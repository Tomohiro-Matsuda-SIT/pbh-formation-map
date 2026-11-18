r"""
Standard (quadratic) curvaton on a fixed Starobinsky background:
theta -> (M_*, f_PBH), prior = Gaussian centred on the observationally allowed region.

=============================== ASSUMPTIONS ===============================
Background inflation (FIXED, not varied):
  * Starobinsky potential V = (3/4) mu^2 M_Pl^2 (1 - exp(-sqrt(2/3) phi/M_Pl))^2, slow-roll H^2 = V/(3 M_Pl^2).
  * mu fixed by A_s = 2.1e-9 at k_* = 0.05 Mpc^-1 (Planck 2018 value, ln(1e10 A_s) = 3.044).
  * Reheating: scalaron -> minimally coupled Higgs, Gamma = mu^3 / (192 pi M_Pl^2);
    rho_reh = 3 M_Pl^2 Gamma^2 ; matter-dominated expansion from end of inflation to reheating.
    (Gives T_reh ~ 3e9 GeV, cf. Bezrukov & Gorbunov arXiv:1111.4397.)
  * N_* solved self-consistently from k_* = a_* H_*.
  * g_rho(T), g_s(T) from the Saikawa-Shirai fitting functions (arXiv:1803.01038, App. C).
Curvaton (VARIED): theta = (m/H_*, log10 sigma_*/H_*, log10 T_dec/GeV)
  * V(sigma) = m^2 sigma^2 / 2, spectator during inflation.
  * Background and perturbation decay per e-fold with lambda(N) = (3/2)[1 - sqrt(1 - 4 m^2 / 9 H(N)^2)]
    (exact growing mode of a massive field for constant H, applied adiabatically).
  * P_x(k) with x = delta sigma / sigma:  P_x = (H(N_k) / (2 pi sigma(N_k)))^2  (massless amplitude at exit;
    O(m^2/H^2) correction to the amplitude ignored).
  * Oscillation energy fraction Omega_osc = sigma_end^2 / (6 M_Pl^2) (fraction frozen during MD reheating),
    grows ~ a after reheating; p = Omega_osc (g_s(T_reh)/g_s(T_dec))^(1/3) T_reh / T_dec,
    Omega_dec = p / (1 + p). Early curvaton domination's effect on N_* and on k<->M is ignored.
  * Sudden decay, fully non-linear delta N (Sasaki, Valiviita, Wands, astro-ph/0607627):
      (1 - Omega) e^{-4 zeta} + Omega (1 + x)^2 e^{-3 zeta} = 1 , inflaton zeta_r neglected on small scales.
  * Only modes super-horizon at curvaton decay (k <= k_dec) acquire zeta; k > k_dec set to zero.
PBH formation (simplified, to be upgraded to compaction function):
  * Threshold on zeta: zeta_c = 1.0 (band 0.7 - 1.3, PLACEHOLDER).
  * beta(M) = Prob(zeta > zeta_c) for Gaussian x with variance s^2(k) = \int dlnq P_x(q) exp(-q^2/k^2).
  * M = gamma M_H(T at re-entry), gamma = 0.2, radiation domination.
  * df/dlnM = beta(M) * rho_tot(T) (a/a_0)^3 / rho_DM,0 ; Omega_DM h^2 = 0.120.
  * M_* = mass at the peak of df/dlnM ; f_PBH = \int dlnM df/dlnM.
Prior:
  * Allowed region in theta from NON-PBH requirements (C1-C5 below); Gaussian with the mean and covariance
    of that region (uniform sampling in the box), truncated to the region.
===========================================================================
"""
from paths import PBH_ROOT, PBHBOUNDS, CEFFYL, NG15, NPROC
import numpy as np
from scipy.special import log_ndtr
from scipy.optimize import brentq

rng = np.random.default_rng(20261113)

# ---------------- constants ----------------
MPl = 2.435e18            # GeV
GeV_to_g = 1.78266e-24
GeV_to_Mpcinv = 1.5637e38
Msun_g = 1.98847e33
T0 = 2.7255 * 8.617333e-14
gs0 = 3.931
As, kstar = 2.1e-9, 0.05
gamma = 0.2
rho_DM0 = 0.120 * 1.05375e-5 * (1.97327e-14)**3

# g_rho(T), g_s(T): Saikawa & Shirai arXiv:1803.01038 App. C fitting functions (gstar_ss.py)
import gstar_ss
def gstar(T):
    """energy-density degrees of freedom g_rho(T)"""
    return gstar_ss.g_rho(T)
def gstar_s(T):
    """entropy degrees of freedom g_s(T)"""
    return gstar_ss.g_s(T)

def H_rad(T):
    return np.sqrt(np.pi**2 / 90 * gstar(T)) * T**2 / MPl

def aH_Mpc(T):
    return (gs0 / gstar_s(T))**(1/3) * T0 / T * H_rad(T) * GeV_to_Mpcinv

# ---------------- Starobinsky background ----------------
yend = np.log(1 + 2 / np.sqrt(3))
def N_of_y(y):
    return 0.75 * (np.exp(y) - y) - 0.75 * (np.exp(yend) - yend)
def y_of_N(N):
    return brentq(lambda y: N_of_y(y) - N, yend, 30)
def V_over(y):   # V / (mu^2 MPl^2)
    return 0.75 * (1 - np.exp(-y))**2
def epsV(y):
    return (4/3) / (np.exp(y) - 1)**2

def solve_background():
    Nst = 55.0
    for _ in range(50):
        y = y_of_N(Nst)
        mu = np.sqrt(As * 24 * np.pi**2 * epsV(y) / V_over(y)) * MPl
        Gam = mu**3 / (192 * np.pi * MPl**2)
        rho_reh = 3 * MPl**2 * Gam**2
        Treh = (rho_reh * 30 / (np.pi**2 * 106.75))**0.25
        rho_end = V_over(yend) * mu**2 * MPl**2 * 1.5     # V/(1 - eps/3) at eps=1 -> rho = 3/2 V
        Hst = np.sqrt(V_over(y) / 3) * mu
        Hend = np.sqrt(rho_end / 3) / MPl
        aHend = aH_Mpc(Treh) * (rho_end / rho_reh)**(1/6)
        Nnew = np.log(aHend / kstar) + np.log(Hst / Hend)
        if abs(Nnew - Nst) < 1e-6:
            break
        Nst = Nnew
    return dict(Nst=Nst, mu=mu, Gam=Gam, Treh=Treh, Hst=Hst, Hend=Hend, aHend=aHend)

BG = solve_background()
Ngrid = np.linspace(0.0, BG['Nst'], 1200)
Hgrid = np.array([np.sqrt(V_over(y_of_N(N)) / 3) * BG['mu'] for N in Ngrid])
# Variable background (2026-11-15): if PBH_BG points to a JSON written by background.write_bg, it replaces the
# fixed Starobinsky background for every module that imports this one (alpha-attractor E/T, free T_reh, w_reh).
import os as _os, json as _json
if _os.environ.get("PBH_BG"):
    _b = _json.load(open(_os.environ["PBH_BG"]))
    Ngrid, Hgrid = np.array(_b.pop("Ngrid")), np.array(_b.pop("Hgrid"))
    BG = _b
lnk_of_N = np.log(BG['aHend']) - Ngrid + np.log(Hgrid / BG['Hend'])   # k = aH at exit

# table T -> (ln k, ln M[Msun], f/beta)
_Ttab = np.logspace(-3.5, np.log10(BG['Treh']), 2000)
_lnk_T = np.log(aH_Mpc(_Ttab))
_M_T = gamma * 4 * np.pi * MPl**2 / H_rad(_Ttab) * GeV_to_g / Msun_g
_rho = np.pi**2 / 30 * gstar(_Ttab) * _Ttab**4
_fb_T = _rho * ((gs0 / gstar_s(_Ttab))**(1/3) * T0 / _Ttab)**3 / rho_DM0
# lnk increases with T
def T_of_lnk(lnk):
    return np.exp(np.interp(lnk, _lnk_T, np.log(_Ttab)))

# ---------------- curvaton map ----------------
def curvaton_background(mH, lsig):
    """returns ln sigma(N) on Ngrid (shape [n, len(Ngrid)]) and validity mask (light throughout)."""
    m = mH[:, None] * BG['Hst']
    r = (4/9) * m**2 / Hgrid[None, :]**2
    light = np.all(r < 1.0, axis=1)
    lam = 1.5 * (1 - np.sqrt(np.clip(1 - r, 0, None)))
    # integrate from N_* downward: ln sigma(N) = ln sigma_* - int_N^{N*} lam dN'
    dN = np.diff(Ngrid)
    cum = np.concatenate([np.zeros((len(mH), 1)),
                          np.cumsum(0.5 * (lam[:, 1:] + lam[:, :-1]) * dN, axis=1)], axis=1)
    cum_from_top = cum[:, -1:] - cum
    lnsig = (np.log(10) * lsig + np.log(BG['Hst']))[:, None] - cum_from_top
    return lnsig, light

def Px_grid(lnsig):
    return (Hgrid[None, :] / (2 * np.pi))**2 * np.exp(-2 * lnsig)

def Omega_dec(lnsig_end, lTdec):
    Tdec = 10**lTdec
    Om_osc = np.exp(2 * lnsig_end) / (6 * MPl**2)
    p = Om_osc * (gstar_s(BG['Treh']) / gstar_s(Tdec))**(1/3) * BG['Treh'] / Tdec
    return p / (1 + p)

def constraints(theta, chunk=20000, relax=()):
    if len(theta) > chunk:
        oks = [constraints(theta[i:i+chunk], relax=relax)[0] for i in range(0, len(theta), chunk)]
        return np.concatenate(oks), None
    mH, lsig, lT = theta.T
    lnsig, light = curvaton_background(mH, lsig)
    Px = Px_grid(lnsig)
    Om = Omega_dec(lnsig[:, 0], lT)
    rr = 3 * Om / (4 - Om)
    Pz_cmb = (2 * rr / 3)**2 * Px[:, -1]
    w = Pz_cmb / (As + Pz_cmb)
    fNL = 5 / (4 * np.maximum(rr, 1e-300)) * w**2
    sig_st = 10**lsig * BG['Hst']
    m = mH * BG['Hst']
    C1 = (Pz_cmb < 0.01 * As) & (np.abs(fNL) < 5.0)              # CMB: amplitude & f_NL^loc
    C2 = 0.5 * m**2 * sig_st**2 < 0.01 * 3 * MPl**2 * BG['Hst']**2   # spectator
    C3 = light                                                   # light during inflation
    C4 = Px.max(axis=1) < 1.0                                    # classical background
    C5 = (10**lT > 5e-3) & (10**lT < BG['Treh'])                 # BBN, decay after reheating
    Cs = dict(C1=C1, C2=C2, C3=C3, C4=C4, C5=C5)
    ok = np.ones(len(theta), bool)
    for k, v in Cs.items():
        if k not in relax:
            ok &= v
    return ok, dict(lnsig=lnsig, Px=Px, Om=Om, **Cs)

def pbh_map(theta, zeta_c=1.0, nk=160, chunk=20000, relax=(), skip=-60):
    """returns log10 M_* [Msun], log10 f_PBH (=-inf if negligible)."""
    if len(theta) > chunk:
        parts = [pbh_map(theta[i:i+chunk], zeta_c, nk, relax=relax, skip=skip) for i in range(0, len(theta), chunk)]
        return tuple(np.concatenate(p) for p in zip(*parts))
    ok, aux = constraints(theta, relax=relax)
    lnsig, Px, Om = aux['lnsig'], aux['Px'], aux['Om']
    lT = theta[:, 2]
    n = len(theta)
    logM = np.full(n, np.nan)
    logf = np.full(n, -np.inf)
    lnk_dec = np.log(aH_Mpc(10**lT))
    y_c = (1 - (1 - Om) * np.exp(-4 * zeta_c)) * np.exp(3 * zeta_c) / Om
    lnkN = lnk_of_N[::-1]          # increasing
    for i in range(n):
        if not ok[i]:
            continue
        Pq = Px[i, ::-1]
        sel = lnkN <= lnk_dec[i]
        if sel.sum() < 3:
            continue
        lq, P = lnkN[sel], Pq[sel]
        # quick upper bound: s^2 <= int P dlnq
        s2max = np.trapezoid(P, lq)
        a = np.sqrt(y_c[i]) - 1
        if (log_ndtr(-a / np.sqrt(s2max)) / np.log(10)) < skip:
            continue
        lk = np.linspace(lnk_dec[i] - 14, lnk_dec[i], nk)
        Wt = np.exp(-np.exp(2 * (lq[None, :] - lk[:, None])))
        s2 = np.trapezoid(P[None, :] * Wt, lq, axis=1)
        s = np.sqrt(s2)
        lnb = np.logaddexp(log_ndtr(-(np.sqrt(y_c[i]) - 1) / s),
                           log_ndtr(-(np.sqrt(y_c[i]) + 1) / s))
        T = T_of_lnk(lk)
        lnM = np.log(np.interp(np.log(T), np.log(_Ttab), _M_T))
        lnfb = np.log(np.interp(np.log(T), np.log(_Ttab), _fb_T))
        lndf = lnb + lnfb                                  # df/dlnM
        j = np.argmax(lndf)
        # integrate over lnM (lnM decreases with lk)
        order = np.argsort(lnM)
        lf = np.log(np.trapezoid(np.exp(lndf[order] - lndf[j]), lnM[order])) + lndf[j]
        logM[i] = lnM[j] / np.log(10)
        logf[i] = lf / np.log(10)
    return ok, logM, logf

# ---------------- prior ----------------
BOX = np.array([[0.05, 1.5], [-1.0, 7.0], [np.log10(5e-3), np.log10(BG['Treh'])]])

def build_prior(nbox=400_000):
    th = BOX[:, 0] + (BOX[:, 1] - BOX[:, 0]) * rng.random((nbox, 3))
    ok, _ = constraints(th)
    allowed = th[ok]
    return allowed.mean(axis=0), np.cov(allowed.T), ok.mean(), allowed

def sample_prior(mean, cov, n):
    out = []
    while sum(len(o) for o in out) < n:
        th = rng.multivariate_normal(mean, cov, size=2 * n)
        inbox = np.all((th >= BOX[:, 0]) & (th <= BOX[:, 1]), axis=1)
        th = th[inbox]
        ok, _ = constraints(th)
        out.append(th[ok])
    return np.concatenate(out)[:n]

if __name__ == "__main__":
    print("Background:", {k: f"{v:.4g}" for k, v in BG.items()})
    mean, cov, acc, allowed = build_prior()
    print("allowed fraction of box:", acc)
    print("prior mean:", mean)
    print("prior std :", np.sqrt(np.diag(cov)))
    print("prior corr:\n", np.corrcoef(allowed.T))
    np.savez(PBH_ROOT + "/prior.npz", mean=mean, cov=cov, box=BOX)
