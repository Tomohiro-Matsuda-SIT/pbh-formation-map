"""
Cosmic-string-loop PBHs on the fixed Starobinsky background.

ASSUMPTIONS (all at top, per project rules)
  Background      : Starobinsky (curvaton_pipeline.BG): H_end, Gamma, T_reh = 2.5e9 GeV fixed.
  String formation: thermal symmetry restoration required, T_c ~ eta <= T_max (max. temperature during
                    perturbative reheating, instant thermalisation assumed  [要確認: Harigaya-Mukaida 1312.3097
                    show delayed thermalisation can lower T_max]).
  Tension         : G mu = 2 pi (eta / M_Pl)^2  (Abelian-Higgs, BPS limit; O(1) convention factor 要確認).
  Loop network    : James-Turner, Weil, Green, Copeland, PRD 101, 123526 (2020) [arXiv:1911.12658]:
                    A = 44, delta = 0.1, alpha = 0.1, 1-<v^2> = 0.6;  M(t) = alpha mu c t  (radiation era).
                    dOmega_eff/dlnM = (c_s/2) (M/M_*)^(-1/2),  M_* = 5e14 g,                       (JT Eq. 23)
                    c_s = 3e2 f (G mu / 1.5e-11)^(3/2)                                              (JT Eq. 14)
                    Loops formed after t_eq (matter era) are NOT included (their dOmega/dlnM is flat in M
                    and suppressed by (M_eq/M_*)^(-1/2) ~ 1e-10; negligible).
  Collapse frac f : (i) 'CC'  : Caldwell & Casper, PRD 53, 3002 (1996) [astro-ph/9509012]
                                f = 10^4.9 (G mu)^4.1  (fitted for 1e-3 < G mu < 3e-2; extrapolated here, 要確認)
                    (ii) 'free': log10 f treated as a free parameter (Gaussian prior on [-30, 0]).
  Evaporated PBHs : evap.py (BBN, CMB distortions/anisotropies, EGB; Carr et al. 2017 integral) applied to the
                    M < 5.1e14 g part of the same MF.  NOTE: our evaluation of JT's own Eqs. (17),(20),(21) gives
                    c_s < 6.3e-11 from CMB anisotropies, not JT's quoted 2e-12 (Eq. 24); an independent re-derivation
                    gives ~5e-11 as well.  JT's EGB-secondary 1e-10 is consistent with ours (5.7e-10 with the review's
                    5e-28 normalisation vs JT's 3e-27).  -> the 2e-12 is 要確認; we use our own evaluation.  Non-evaporated part (M > 5.1e14 g):
                    PBHbounds envelopes with the Carr et al. (2017) extended-MF method (emf.py).
  GW bound on Gmu : LVK O3, PRL 126, 241102 (2021) [arXiv:2101.12248]: G mu < 4e-15 (most stringent loop
                    model; weaker for others).  PTA bounds (~1e-10) irrelevant here.
  Omega_DM h^2    : 0.120, h = 0.674 (Planck 2018).
"""
import numpy as np
from scipy.integrate import solve_ivp
import curvaton_pipeline as cp
from gstar_ss import g_rho
import emf
import evap

MPr = 2.435e18                      # reduced Planck mass [GeV]
MPl = 1.2209e19                     # Planck mass [GeV]
C3G = 4.0370e38                     # c^3/G [g/s]
MSUN = 1.98847e33
MSTAR = 5e14                        # g
T_EQ = 1.7e12                       # s (JT)
OM_DM = 0.120 / 0.674**2
ALPHA = 0.1
M_EVAP = 5.1e14                     # g

def T_max():
    """Max. radiation temperature during perturbative reheating (Giudice-Kolb-Riotto hep-ph/0005123 setup)."""
    if "Tmax" in cp.BG:          # variable background (background.py): w_reh-dependent T_max already computed
        return cp.BG["Tmax"], np.nan
    H0, G = cp.BG["Hend"], cp.BG["Gam"]
    def rhs(x, y):
        rp, rr = np.exp(y)
        H = np.sqrt((rp + rr) / 3) / MPr
        return [-3 - G / H, -4 + G / H * rp / rr]
    y0 = [np.log(3 * MPr**2 * H0**2), np.log(3 * MPr**2 * H0**2 * 1e-30)]
    sol = solve_ivp(rhs, [0, 40], y0, rtol=1e-8, atol=1e-10, dense_output=True)
    x = np.linspace(0, 40, 40001)
    rr = np.exp(sol.sol(x)[1])
    T = (30 * rr / (np.pi**2 * 106.75))**0.25
    for _ in range(3):
        T = (30 * rr / (np.pi**2 * g_rho(T)))**0.25
    i = np.argmax(T)
    return T[i], x[i]

def gmu_of_eta(eta):
    return 2 * np.pi * (eta / MPl)**2

def c_string(lf, lgmu):
    return 3e2 * 10**lf * (10**lgmu / 1.5e-11)**1.5

def f_CC(lgmu):
    return 4.9 + 4.1 * lgmu

LM = np.linspace(np.log10(M_EVAP), np.log10(2e41), 400)     # g grid for the non-evaporated MF

LME = np.linspace(8, np.log10(M_EVAP), 600)                  # g grid for the evaporated part (limits start 1e8 g)

def mf(lc, lgmu):
    """dfPBH/dlnM on grid LM (grams) for log10 c_s = lc; cut at M_eq = alpha mu c t_eq."""
    Meq = ALPHA * 10**lgmu * C3G * T_EQ
    psi = 0.5 * 10**lc * (10**LM / MSTAR)**-0.5 / OM_DM
    return np.where(10**LM <= Meq, psi, 0.0)

def flags(lc, lgmu):
    """returns (ftot, ok, det); emf works in log10(M/Msun)."""
    lMs = LM - np.log10(MSUN)
    psi = mf(lc, lgmu)
    ftot = np.trapezoid(psi, LM * np.log(10))
    cur = emf.integrals(lMs, psi, emf.CURRENT)
    fut = emf.integrals(lMs, psi, emf.FUTURE)
    ok = (max(cur.values()) <= 1) and bool(evap.ok(LME, 0.5 * 10**lc * (10**LME / MSTAR)**-0.5 / OM_DM)[0])
    det = max(fut.values()) > 1
    return ftot, ok, det

if __name__ == "__main__":
    Tm, xm = T_max()
    gmax_T = gmu_of_eta(Tm)
    print(f"T_max = {Tm:.3e} GeV at N = {xm:.2f} after end of inflation;  T_max/T_reh = {Tm/cp.BG['Treh']:.1f}")
    print(f"G mu (eta = T_max) = {gmax_T:.2e};  (eta = T_reh) = {gmu_of_eta(cp.BG['Treh']):.2e};  LVK: 4e-15")
    lg_hi = np.log10(min(gmax_T, 4e-15)); lg_lo = -26.0
    # max non-evaporated abundance allowed by the CMB bound alone
    for lg in (lg_lo, -20, lg_hi):
        cmax = 1 / max(evap.integrals(LME, 0.5 * (10**LME / MSTAR)**-0.5 / OM_DM).values())
        print(f"  lgmu={lg:.1f}: evap -> c_s<={cmax:.2e} -> f_tot <= {flags(np.log10(cmax), lg)[0]:.2e}; "
              f"needs log f <= {np.log10(cmax) - np.log10(c_string(0, lg)):.1f}; CC gives log f = {f_CC(lg):.1f}, "
              f"log c_s(CC) = {np.log10(c_string(f_CC(lg), lg)):.1f}")
    # cusp collapse (Jenkins-Sakellariadou 2006.16249): M_PBH ~ (G mu)^2 M_loop, M_loop <= alpha mu c t_0
    t0 = 4.35e17
    print(f"  cusp-PBH max mass (lgmu={lg_hi:.1f}): {(10**lg_hi)**2 * ALPHA * 10**lg_hi * C3G * t0:.1e} g  (< M_evap)")
    # Gaussian prior on the allowed box (project rule): log Gmu in [lg_lo, lg_hi], log f in [-30, 0] (free)
    rng = np.random.default_rng(7)
    out = {}
    for tag in ("CC", "free"):
        n = 20000
        mu_, sd_ = 0.5 * (lg_lo + lg_hi), (lg_hi - lg_lo) / np.sqrt(12)
        lg = rng.normal(mu_, sd_, 4 * n); lg = lg[(lg > lg_lo) & (lg < lg_hi)][:n]
        if tag == "CC":
            lf = f_CC(lg)
        else:
            lf = rng.normal(-15, 30 / np.sqrt(12), 4 * n); lf = lf[(lf > -30) & (lf < 0)][:n]
        lc = np.log10(c_string(lf, lg))
        res = np.array([flags(a, b) for a, b in zip(lc, lg)])
        ftot, ok, det = res.T
        lMrep = np.full(n, np.log10(MSTAR / MSUN))        # abundance dominated by the lightest surviving PBHs
        out[tag] = np.array([lMrep, np.log10(np.maximum(ftot, 1e-300)), ok, det])
        print(f"[{tag}] P(ok)={ok.mean():.3f}  P(f_tot>1e-12)={(ftot > 1e-12).mean():.4f} "
              f"P(f_tot>1e-12 & ok)={((ftot > 1e-12) & (ok > 0)).mean():.4f}  P(det&ok)={((det > 0) & (ok > 0)).mean():.4f}"
              f"  max f_tot|ok = {ftot[ok > 0].max():.2e}")
    np.savez("cstring_samples.npz", **out, Tmax=Tm, lg_hi=lg_hi)
