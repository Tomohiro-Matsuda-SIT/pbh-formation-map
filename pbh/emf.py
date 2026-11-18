r"""
Observational constraints applied to extended PBH mass functions.

Method (Carr, Raidal, Tenkanen, Vaskonen, Veermae, PRD 96 (2017) 023514, arXiv:1705.05567):
   a mass function psi(M) (df/dlnM, integrating to f_PBH) is excluded by constraint c if
        I_c = \int dlnM psi(M) / f_max,c(M) > 1 ,
   with f_max,c the monochromatic bound (infinite outside its mass range).
Constraints are applied per PBHbounds category envelope (Evaporation, Microlensing, GWs, Accretion, Dynamical).
Using category envelopes (rather than each individual bound) is slightly more aggressive than bound-by-bound;
noted as a systematic.  Projected probes are applied bound by bound.

Mass functions used here
   * FOPT (GV, non-covariant): critical collapse  M = K M_H Delta^gamma, Delta = delta - delta_c,
       K = 3, gamma = 0.36 (as in FGJ arXiv:2503.01962 Eq. 20), M_H = 4.8e-2 M_sun (106.75/g)^{1/2} (GeV/T)^2
       (FGJ Eq. 23).  Delta distributed exponentially with scale s = -1/(d ln P_coll/d delta_c) from the GV fit
       (rare-event tail).  Mass-weighted shape: psi(u) ∝ u^{1+gamma} e^{-u}, u = Delta/s, lnM = ln(K M_H s^gamma) + gamma ln u.
       Total f_PBH kept equal to the GV normalisation; M_* = peak of psi = K M_H ((1+gamma) s)^gamma.
   * DW: monochromatic at M = t_dom/(2G) (GV); extended shape not modelled (TO DO).
"""
from paths import PBH_ROOT, PBHBOUNDS, CEFFYL, NG15, NPROC
import numpy as np

PB = PBHBOUNDS + "/bounds/"
CURRENT = ["Evaporation", "Microlensing", "GWs", "Accretion", "Dynamical"]
FUTURE = ["3G-GW-10y", "GECCO", "Xraylensing-Xmu-proj", "Xraylensing-eXTP-proj", "WhiteDwarfmicro",
          "PTA-Doppler", "PTA-Shapiro", "SKA", "GRB-parallax", "FL-small"]

def _load(name):
    d = np.loadtxt(PB + name + ".txt", comments="#", usecols=(0, 1))
    d = d[np.isfinite(d).all(1) & (d[:, 0] > 0) & (d[:, 1] > 0)]
    o = np.argsort(d[:, 0])
    return np.log10(d[o, 0]), np.log10(d[o, 1])

BOUNDS = {n: _load(n) for n in CURRENT + FUTURE}

def log_fmax(name, logM):
    lm, lf = BOUNDS[name]
    out = np.interp(logM, lm, lf)
    return np.where((logM >= lm[0]) & (logM <= lm[-1]), out, np.inf)

def integrals(logM_grid, psi, names):
    """psi: (n, m) df/dlnM on logM_grid (n samples, m grid points, may differ per sample: logM_grid (n, m)).
       returns dict name -> I (n,)"""
    lnM = logM_grid * np.log(10)
    out = {}
    for nm in names:
        inv = 10**(-log_fmax(nm, logM_grid))          # 0 outside range
        out[nm] = np.trapezoid(psi * inv, lnM, axis=-1)
    return out

GAMMA, KCC = 0.36, 3.0
U = np.logspace(-3, np.log10(30), 240)                 # u = Delta/s
_W = U**(1 + GAMMA) * np.exp(-U)
_W_lnM = _W / np.trapezoid(_W, GAMMA * np.log(U))       # normalised in lnM (dlnM = gamma dln u)

def fopt_emf(logf, logMH, s):
    """critical-collapse mass function for FOPT samples.
       returns logM grid (n, len(U)), psi (n, len(U)), logM_star (n,)"""
    logM = (np.log10(KCC) + logMH + GAMMA * np.log10(s))[:, None] + GAMMA * np.log10(U)[None, :]
    psi = 10**logf[:, None] * _W_lnM[None, :]
    logMstar = np.log10(KCC) + logMH + GAMMA * np.log10((1 + GAMMA) * s)
    return logM, psi, logMstar

def mono_flags(logM, logf, names):
    """monochromatic: excluded by constraint if f > f_max(M)."""
    return {nm: logf > log_fmax(nm, logM) for nm in names}
