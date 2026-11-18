r"""
PBHs from spherical domain walls nucleated during (Starobinsky) inflation
(Garriga, Vilenkin & Zhang, arXiv:1512.01819; Deng, Garriga & Vilenkin, arXiv:1612.03753).

=============================== ASSUMPTIONS ===============================
Nucleation: rate per Hubble volume per Hubble time lambda(N) = A exp[-S_E(N)], S_E = 2 pi^2 sigma / H(N)^3
   (DGV: lambda ∝ e^{-S_E}, S_E ≈ 2 pi^2 sigma H_i^{-3}); H(N) is the FIXED Starobinsky H(N); A = O(1) prefactor
   (unknown; band A = 1e-2 / 1 / 1e2).
Size distribution: dn = lambda dR / R^4 (DGV Eq. 3). A wall nucleated N e-folds before the end of inflation
   (Hubble size then) is stretched with the comoving scale k(N) = a H at exit; it re-enters the horizon in the
   radiation era at t_H with k = a H.  At re-entry: dn/dlnR = lambda / (2 t_H)^3 (physical, per log size).
BH mass (interpolating sub-/supercritical; DGV: M_crit ~ t_sigma/G, supercritical M ~ t_H/G, accretion ~ x2):
   M = kappa (t_H/G) * x/(1+x),  x = 8 t_H / (kappa t_sigma),  t_sigma = 1/(2 pi G sigma);
   subcritical limit M -> 16 pi sigma t_H^2 (wall area at horizon crossing, radius 2 t_H); kappa = 2 (accretion).
Energy fraction per lnR at formation: beta_lnR = M (dn/dlnR) / rho_r(t_H);  df/dlnM = (f/beta)(T_H) beta_lnR / (dlnM/dlnR).
   (f/beta)(T) = rho_tot(T) (a/a0)^3 / rho_DM0 (curvaton_pipeline._fb_T). Reference: only re-entry in radiation
   domination after reheating (T_H < T_reh) is kept; walls are assumed to interact with matter only gravitationally.
   Alternative (env PBH_IW_REH=collapse): walls re-entering during the reheating era also collapse (see _setup).
Parameter: x = log10(sigma^{1/3}/H_*).  Allowed region (non-PBH): dilute nucleation lambda(N) <= 0.1 and S_E >= 1 for all N
   (needed for the isolated-wall picture; otherwise a wall network forms), sigma^{1/3} <= M_P.
   => box x in [log10 of S_E(N_*) = 2.3 + ln A, log10(M_P/H_*)]; the upper edge (M_P) is a THEORETICAL choice, and it
   largely sets the prior (flagged).
Mass function: broad, evaluated with the Carr et al. extended-MF method (emf.py). Only M > 5.1e14 g counted today;
   limits on already-evaporated PBHs (M < 5.1e14 g; BBN, CMB distortions/anisotropies, EGB) applied with evap.py
   (2026-11-14; previously omitted).
Bands: (A, kappa) = optimistic (1e2, 2), central (1, 2), conservative (1e-2, 1).
===========================================================================
"""
from paths import PBH_ROOT, PBHBOUNDS, CEFFYL, NG15, NPROC
import numpy as np
import curvaton_pipeline as cp
import emf
import evap

rng = np.random.default_rng(202)
MP = cp.MPl
G = 1 / (8 * np.pi * MP**2)
GeV_to_g, Msun_g = 1.78266e-24, 1.98847e33

Ng, Hg, lnk = cp.Ngrid, cp.Hgrid, cp.lnk_of_N        # N ascending (0 = end), lnk descending
T_H = cp.T_of_lnk(lnk)                                 # re-entry temperature (clamped at T_reh for earlier re-entry)

# Re-entry BEFORE reheating (2026-11-16, Kohri-consistency review C1).  Reference treatment REH="drop": such walls are
# discarded (the docstring assumption).  Alternative REH="collapse": they collapse during the reheating era with constant
# equation of state w = BG['w'] (w = 0 for scalaron reheating), where pressure does not hinder collapse
# (Harada, Yoo, Kohri, Nakao, Jhingan arXiv:1609.01588).  Horizon re-entry k = aH with aH ∝ a^{-(1+3w)/2},
# H ∝ a^{-3(1+w)/2} (H_reh = H_rad(T_reh)); the same horizon-scale relations are used with t_H ≡ 1/(2H) and
# rho = 3 M_P^2 H^2, and the PBH energy fraction grows as a^{3w} until reheating, so f/beta = (f/beta)(T_reh)(a_reh/a)^{3w}.
import os as _os
REH = _os.environ.get("PBH_IW_REH", "drop")
def _setup(reh):
    lnk_reh = np.log(cp.aH_Mpc(cp.BG["Treh"]))
    before = lnk > lnk_reh
    H = cp.H_rad(T_H); rho = np.pi**2 / 30 * cp.gstar(T_H) * T_H**4
    f_b = np.interp(np.log(T_H), np.log(cp._Ttab), cp._fb_T)
    if reh == "collapse" and before.any():
        w = float(cp.BG.get("w", 0.0))
        lna = -2 / (1 + 3 * w) * (lnk - lnk_reh)                     # ln(a/a_reh) < 0
        Hreh = cp.H_rad(cp.BG["Treh"])
        H = np.where(before, Hreh * np.exp(-1.5 * (1 + w) * lna), H)
        rho = np.where(before, 3 * MP**2 * H**2, rho)
        f_b = np.where(before, np.interp(np.log(cp.BG["Treh"]), np.log(cp._Ttab), cp._fb_T) * np.exp(-3 * w * lna), f_b)
        kp = T_H > 1e-3
    else:
        kp = (~before) & (T_H < cp.BG["Treh"]) & (T_H > 1e-3)
    return kp, 1 / (2 * H), rho, f_b
keep, t_H, rho_r, fb = _setup(REH)

def wall_mf(x, A=1.0, kappa=2.0):
    """x = log10(sigma^{1/3}/H_*) (array n). returns logM grid (n, m), psi = df/dlnM (n, m), f_tot, logM_peak."""
    sig = (10**x[:, None] * cp.BG["Hst"])**3
    S = 2 * np.pi**2 * sig / Hg[None, :]**3
    lam = A * np.exp(-S)
    tsig = 1 / (2 * np.pi * G * sig)
    xx = 8 * t_H[None, :] / (kappa * tsig)
    M = kappa * t_H[None, :] / G * xx / (1 + xx)                     # GeV
    dndlnR = lam / (2 * t_H[None, :])**3
    beta = M * dndlnR / rho_r[None, :]
    lnR = -lnk                                                        # comoving size ∝ 1/k
    lnM = np.log(M)
    dlnM_dlnR = np.gradient(lnM, lnR, axis=1)
    psi = fb[None, :] * beta / np.maximum(dlnM_dlnR, 1e-3)
    psi_eff = np.where(keep[None, :], psi, 0.0)                              # as if no evaporation
    psi = np.where(keep[None, :] & (M * GeV_to_g > 5.1e14), psi, 0.0)    # evaporated PBHs do not count today
    logM = np.log10(M * GeV_to_g / Msun_g)
    o = np.argsort(logM, axis=1)
    lM_s = np.take_along_axis(logM, o, 1); psi_s = np.take_along_axis(psi, o, 1)
    psie_s = np.take_along_axis(psi_eff, o, 1)
    wall_mf.evap_ok = evap.ok(lM_s + np.log10(Msun_g), np.where(lM_s + np.log10(Msun_g) < np.log10(5.1e14), psie_s, 0.0))[0]
    ftot = np.trapezoid(psi_s, lM_s * np.log(10), axis=1)
    jmax = np.argmax(psi_s, axis=1)
    lMpk = lM_s[np.arange(len(x)), jmax]
    return lM_s, psi_s, ftot, lMpk

def box(A):
    Smin = max(np.log(10 * A), 1.0)            # lambda <= 0.1 (dilute) and S_E >= 1 (semiclassical)
    lo = np.log10(((Smin / (2 * np.pi**2)))**(1 / 3) * (Hg.max() / cp.BG["Hst"]))
    return np.array([lo, np.log10(MP / cp.BG["Hst"])])

def run(A, kappa, n=100_000):
    b = box(A)
    # allowed region = whole box (dilute nucleation is the lower edge); Gaussian with box mean/std, truncated
    u = b[0] + (b[1] - b[0]) * rng.random(400_000)
    mu, sd = u.mean(), u.std()
    x = rng.normal(mu, sd, 3 * n); x = x[(x >= b[0]) & (x <= b[1])][:n]
    out = []; nev = []
    for i in range(0, len(x), 1500):
        xc = x[i:i + 1500]
        lM, psi, ftot, lMpk = wall_mf(xc, A, kappa)
        logf = np.where(ftot > 0, np.log10(np.maximum(ftot, 1e-300)), -np.inf)
        sel = logf > -12
        ok = wall_mf.evap_ok.copy(); det = np.zeros(len(xc), bool)
        if sel.any():
            Ic = emf.integrals(lM[sel], psi[sel], emf.CURRENT); If = emf.integrals(lM[sel], psi[sel], emf.FUTURE)
            ok[sel] &= np.all(np.array([Ic[k] <= 1 for k in Ic]), axis=0)
            det[sel] = np.any(np.array([If[k] > 1 for k in If]), axis=0)
        out.append((lMpk, logf, ok, det))
        nev.append(wall_mf.evap_ok)
    lMpk, logf, ok, det = (np.concatenate(z) for z in zip(*out))
    run.evap_ok = np.concatenate(nev)
    return x, lMpk, logf, ok, det, b

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    res = {}
    for tag, A, kap in [("opt", 1e2, 2.0), ("cen", 1.0, 2.0), ("con", 1e-2, 1.0)]:
        x, lMpk, logf, ok, det, b = run(A, kap)
        sel = logf > -12
        print(f"[{tag}] box x=[{b[0]:.3f},{b[1]:.3f}]  P(f>1e-12)={np.mean(sel):.4f}  P(1e-12<f<1)={np.mean(sel & (logf < 0)):.4f}"
              f"  P(evap-ok)={run.evap_ok.mean():.3f} P(f>1e-12 & evap-ok)={np.mean(sel & run.evap_ok):.4f}"
              f"  P(f>1e-12 & allowed)={np.mean(sel & ok):.4f}  P(allowed & detectable)={np.mean(sel & ok & det):.4f}")
        if sel.any():
            print("     sigma^{1/3}/H_* range with f>1e-12:", np.round(10**np.percentile(x[sel], [0, 50, 100]), 3),
                  " logM_peak 5-50-95%:", np.round(np.percentile(lMpk[sel], [5, 50, 95]), 2))
        res[tag] = np.vstack([lMpk, logf, ok, det])
    np.savez(PBH_ROOT + "/infwall_samples.npz", **res)
