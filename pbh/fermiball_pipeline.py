r"""
PBHs from Fermi-ball collapse after a first-order phase transition (Kawana & Xie, arXiv:2106.00111,
PLB 824 (2022) 136791), on the fixed Starobinsky background.

=============================== ASSUMPTIONS ===============================
Mechanism: a fermion chi that is heavy in the true vacuum (M_chi*/T_* ~ 12) is trapped in the false vacuum
  (trapping fraction F = 0.98, KX benchmark with v_b = 0.6), compressed into false-vacuum remnants -> Fermi-balls,
  which collapse to PBHs once the Yukawa attraction dominates (collapse condition, KX Eq. 16, assumed satisfied by
  the mediator mass; T_phi ~ T_* taken).
Parameters: theta = (log10 T_*/GeV, log10 beta/H, log10 alpha, log10 eta_chi), v_b = 0.6 fixed.
  M_PBH = 1.4e21 g v_b^3 (eta/1e-3)(100/g)^{1/4}(100 GeV/T)^2 (100/(beta/H))^3 alpha^{1/4}   (KX Eq. 14)
  f_PBH = 1.3e3 v_b^{-3} (g/100)^{1/2} (T/100 GeV)^3 ((beta/H)/100)^3 (M_PBH/1e15 g)         (KX Eq. 18)
        => f_PBH = 1.82e10 (eta T/GeV)(g/100)^{1/4} alpha^{1/4} : linear in eta, independent of beta/H.
  g = g_rho(T_*) (Saikawa-Shirai).  Checked: KX benchmark (T=1.1 GeV, g=10, beta/H=100, alpha=1, eta=0.9e-10)
  -> M = 4.0e17 g, f = 1.06 (KX quote 4.2e17 g, 1).
  PBHs with M < 5.1e14 g have evaporated today (f counted as 0 in the (M,f) plane).
  Mass function: monochromatic (remnant-size distribution not modelled).
Non-PBH constraints:
  * GW from the FOPT (sound waves; Caprini et al. arXiv:1512.06239, with the finite-lifetime factor):
      h^2 Omega = 2.65e-6 (H/beta)(kappa alpha/(1+alpha))^2 (100/g_s)^{1/3} v_w S_sw Upsilon,
      kappa = alpha/(0.73+0.083 sqrt(alpha)+alpha), S_sw = x^3 (7/(4+3x^2))^{7/2},
      f_sw = 1.9e-5 Hz (1/v_w)(beta/H)(T/100 GeV)(g_s/100)^{1/6},
      Upsilon = 1 - 1/sqrt(1 + 2 tau H), tau H = (8 pi)^{1/3} v_w / [(beta/H) U], U^2 = 3 kappa alpha / (4(1+alpha)).
    LVK O3 and NANOGrav-upper-limit as in fopt_dw_pipeline.
  * Untrapped chi relic (fraction 1-F = 0.02, mass 12 T_*): Omega_chi h^2 = 0.02 * eta * 12 T * s0/rho_c <= 0.12.
  * T_* in (5 MeV, T_reh).
Box: log T in [log 5e-3, log T_reh], log beta/H in [1, 4], log alpha in [-2, 1], log eta in [-16, -6].
  The lower edge of eta is NOT set by any observation -> prior centre depends on it (arbitrariness, flagged).
===========================================================================
"""
from paths import PBH_ROOT, PBHBOUNDS, CEFFYL, NG15, NPROC
import numpy as np
import curvaton_pipeline as cp
import fopt_dw_pipeline as fd
import emf

rng = np.random.default_rng(101)
vb = 0.6
TREH = cp.BG["Treh"]
s0 = 2891.2                       # cm^-3
rhoc_h2 = 1.05375e-5              # GeV cm^-3
BOX = np.array([[np.log10(5e-3), np.log10(TREH)], [1.0, 4.0], [-2.0, 1.0], [-16.0, -6.0]])

def fb_gw(lT, lbH, la):
    T, bH, a = 10**lT, 10**lbH, 10**la
    gs = cp.gstar_s(T)
    kap = a / (0.73 + 0.083 * np.sqrt(a) + a)
    U = np.sqrt(3 * kap * a / (4 * (1 + a)))
    tauH = (8 * np.pi)**(1 / 3) * vb / (bH * U)
    Ups = 1 - 1 / np.sqrt(1 + 2 * tauH)
    fsw = 1.9e-5 / vb * bH * (T / 100) * (gs / 100)**(1 / 6)
    amp = 2.65e-6 / bH * (kap * a / (1 + a))**2 * (100 / gs)**(1 / 3) * vb * Ups / fd.h**2
    def Om(f):
        x = f[None, :] / fsw[:, None]
        return amp[:, None] * x**3 * (7 / (4 + 3 * x**2))**3.5
    return Om

def fb_map(P):
    lT, lbH, la, le = P.T
    T, bH, a, eta = 10**lT, 10**lbH, 10**la, 10**le
    g = cp.gstar(T)
    M_g = 1.4e21 * vb**3 * (eta / 1e-3) * (100 / g)**0.25 * (100 / T)**2 * (100 / bH)**3 * a**0.25
    f = 1.3e3 * vb**-3 * (g / 100)**0.5 * (T / 100)**3 * (bH / 100)**3 * (M_g / 1e15)
    logM = np.log10(M_g / 1.98847e33)
    logf = np.where(M_g > 5.1e14, np.log10(f), -np.inf)
    return logM, logf

def fb_allowed(P):
    lT, lbH, la, le = P.T
    T, eta = 10**lT, 10**le
    gw = fd.gw_ok(fb_gw(lT, lbH, la))
    relic = 0.02 * eta * 12 * T * s0 / rhoc_h2 <= 0.12
    return (T > 5e-3) & (T < TREH) & gw & relic

def check_benchmark():
    # KX benchmark uses g = 10 explicitly; reproduce with g fixed
    T, bH, a, eta, g = 1.1, 100.0, 1.0, 0.9e-10, 10.0
    M = 1.4e21 * vb**3 * (eta / 1e-3) * (100 / g)**0.25 * (100 / T)**2 * (100 / bH)**3 * a**0.25
    f = 1.3e3 * vb**-3 * (g / 100)**0.5 * (T / 100)**3 * (bH / 100)**3 * (M / 1e15)
    return M, f

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    print("KX benchmark check: M = %.3g g, f = %.3g" % check_benchmark())
    mu, cov, acc, S, A = fd.gaussian_prior(BOX, fb_allowed)
    print(f"allowed fraction {acc:.3f}; prior mean {np.round(mu, 2)}, std {np.round(np.sqrt(np.diag(cov)), 2)}")
    lM, lf = fb_map(S)
    cur = emf.mono_flags(lM, lf, emf.CURRENT); fut = emf.mono_flags(lM, lf, emf.FUTURE)
    ok = ~np.any(np.array(list(cur.values())), axis=0); det = np.any(np.array(list(fut.values())), axis=0)
    sel = lf > -12
    print(f"P(evaporated) = {np.mean(~np.isfinite(lf)):.3f}; P(f>1e-12) = {np.mean(sel):.3f}; "
          f"P(1e-12<f<1) = {np.mean(sel & (lf < 0)):.3f}; P(f>1e-12 & allowed) = {np.mean(sel & ok):.3f}; "
          f"P(allowed & detectable) = {np.mean(sel & ok & det):.3f}")
    print("logM 5-50-95% (f>1e-12):", np.round(np.percentile(lM[sel], [5, 50, 95]), 2))
    print("logf 5-50-95% (f>1e-12):", np.round(np.percentile(lf[sel], [5, 50, 95]), 2))
    # sensitivity to the arbitrary lower edge of eta
    for lo in (-14.0, -18.0):
        B2 = BOX.copy(); B2[3, 0] = lo
        _, _, _, S2, _ = fd.gaussian_prior(B2, fb_allowed, nsamp=200_000)
        lM2, lf2 = fb_map(S2); s2 = lf2 > -12
        cur2 = emf.mono_flags(lM2, lf2, emf.CURRENT); ok2 = ~np.any(np.array(list(cur2.values())), axis=0)
        print(f"  eta lower edge 1e{int(lo)}: P(f>1e-12)={np.mean(s2):.3f}, P(f>1e-12 & allowed)={np.mean(s2 & ok2):.3f}")
    np.savez(PBH_ROOT + "/fermiball_samples.npz", S=S, x=np.vstack([lM, lf, ok, det]))
