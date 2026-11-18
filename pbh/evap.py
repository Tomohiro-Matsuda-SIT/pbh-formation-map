"""
Limits on PBHs that have ALREADY EVAPORATED (M < 5.1e14 g), applied to extended mass functions.

ASSUMPTIONS / SOURCES (beta' = gamma^{1/2} (g_*i/106.75)^{-1/4} beta, Carr et al. 2010 Eq. 2.7)
  Conversion to an 'effective' f (abundance the PBHs would have today had they not evaporated):
     Omega_PBH = (beta'/1.15e-8) (h/0.72)^-2 (M/Msun)^-1/2   [Carr, Kohri, Sendouda, Yokoyama 2010,
                                                               arXiv:0912.5297, Eqs. 2.5, 2.7],
     f_eff = Omega_PBH / Omega_DM,  h = 0.674, Omega_DM h^2 = 0.120.
     (equivalent to the project's f = 1.68e8 beta (gamma/0.2)^{1/2}(g/106.75)^{-1/4}(M/Msun)^{-1/2})
  Upper limits on beta'(M) used (each applied separately, Carr et al. 2017 integral I_c = ∫ dlnM psi/f_max,c <= 1):
   'BBN-KY'  Kohri & Yokoyama 1999 [astro-ph/9908160] (hadron injection):
               beta < 1e-20 for 1e8 < M/g < 1e10;  beta < 1e-22 for 1e10 < M/g < 3e10
               (KY define beta at M = M_H, i.e. gamma = 1, so beta' = beta up to (g_*/106.75)^{-1/4} = O(1); resolved
               2026-11-16).  The updated BBN limits of Carr, Kohri, Sendouda, Yokoyama 2010 (arXiv:0912.5297, Sec. 3) are
               much stronger in 1e9-1e13 g; not digitized.  Sensitivity: adding beta' < 1e-26 over 1e9-1e13 g gives
               c_s < 4.3e-10 for the M^{-1/2} string/wall MF, still weaker than CMB-ani (6.3e-11) -> no change.
   'CMB-mu'  CKSY 2021 review [arXiv:2002.12778] Eq. 25 (spectral distortions):
               beta' < 1e-16 (M/1e11 g)^-1,  1e11 < M/g < 1e13
   'CMB-ani' CKSY 2021 Eq. 28 (anisotropy damping):  beta' < 3e-30 (f_H/0.1)^-1 (M/1e13 g)^3.1, 2.5e13 < M/g < 2.4e14
   'EGB'     CKSY 2021 Eq. 32 (extragalactic gamma, secondary): beta' < 5e-28 (M/M*)^(-5/2-2 eps), eps = 0.1-0.4;
               we use the exponent -2.9, i.e. eps = 0.2 (corrected description 2026-11-16; earlier text said -5/2-eps),
               applied for 1e13 < M/g < M* = 5.1e14 g
   'floor'   CKSY 2021 Sec. I: beta must be < 1e-18 over the whole range 1e9 - 1e15 g (conservative floor;
               fills the 3e10 - 1e11 g gap where the full BBN curve, only numerical, is much stronger ~1e-24).
  => CONSERVATIVE implementation: the numerical BBN curves (CKSY Fig. 5; Keith et al. 2020) are stronger in
     3e10 - 1e13 g; not used because no machine-readable data are available here (flagged).
"""
import numpy as np

MSUN = 1.98847e33
H = 0.674
OM_DM = 0.120 / H**2
MSTAR = 5.1e14

def _seg(lo, hi, fn):
    return (np.log10(lo), np.log10(hi), fn)

LIMITS = {
    "BBN-KY":  [_seg(1e8, 1e10, lambda M: 1e-20 + 0 * M), _seg(1e10, 3e10, lambda M: 1e-22 + 0 * M)],
    "CMB-mu":  [_seg(1e11, 1e13, lambda M: 1e-16 * (M / 1e11)**-1)],
    "CMB-ani": [_seg(2.5e13, 2.4e14, lambda M: 3e-30 * (M / 1e13)**3.1)],
    "EGB":     [_seg(1e13, MSTAR, lambda M: 5e-28 * (M / MSTAR)**(-2.5 - 0.4))],
    "floor":   [_seg(1e9, 1e15, lambda M: 1e-18 + 0 * M)],
}

def beta_to_feff(beta_p, M_g):
    return beta_p / 1.15e-8 * (H / 0.72)**-2 * (M_g / MSUN)**-0.5 / OM_DM

def log_feff_max(name, logM_g):
    out = np.full(np.shape(logM_g), np.inf)
    M = 10**np.asarray(logM_g, float)
    for lo, hi, fn in LIMITS[name]:
        m = (logM_g >= lo) & (logM_g <= hi)
        out = np.where(m, np.minimum(out, np.log10(beta_to_feff(fn(M), M))), out)
    return out

def integrals(logM_g, psi_eff):
    """logM_g: log10(M/g) grid (n,m) or (m,); psi_eff = d f_eff / d lnM (as if no evaporation). -> dict name -> I"""
    lnM = logM_g * np.log(10)
    return {nm: np.trapezoid(psi_eff * 10**(-log_feff_max(nm, logM_g)), lnM, axis=-1) for nm in LIMITS}

def ok(logM_g, psi_eff):
    I = integrals(logM_g, psi_eff)
    return np.all(np.array([I[k] <= 1 for k in I]), axis=0), I

if __name__ == "__main__":
    # check against James-Turner et al. (2020): cosmic-string MF dOmega/dlnM = (c/2)(M/5e14 g)^-1/2 -> c < 2e-12
    lM = np.linspace(8, np.log10(5.1e14), 2000)
    psi1 = 0.5 * (10**lM / 5e14)**-0.5 / OM_DM            # c = 1
    I = integrals(lM, psi1)
    for k, v in I.items():
        print(f"{k:8s}  c_max = {1 / v:.2e}" if v > 0 else k)
    print("monochromatic f_eff max at 1e10, 1e12, 1e14 g:",
          [f"{10**min(log_feff_max(k, np.array(l)) for k in LIMITS):.1e}" for l in (10., 12., 14.)])
