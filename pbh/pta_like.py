"""
PTA free-spectrum likelihood (replaces the single-frequency 1/yr condition for 'DW explains PTA').

DATA (provenance)
  KDE representations of Hellings-Downs-correlated free spectra distributed with ceffyl
  (github.com/astrolamb/ceffyl, commit 0cd2b3a; Lamb, Taylor & van Haasteren, PRD 108, 103019 (2023),
  arXiv:2303.15442):
     'EPTA'  data/epta_dr2_new_hd  (EPTA DR2new, 20 frequencies, f_1 = 2.87 nHz)
     'PPTA'  data/ppta_dr3_hd      (PPTA DR3, 40 frequencies, f_1 = 1.68 nHz)
     'NG15'  NANOGrav 15-yr HD free spectrum (30f_fs{hd}_ceffyl, Zenodo 10.5281/zenodo.10344086 v1.0.0,
             supplied by the user 2026-11-14), first 14 frequencies (f_1 = 1.98 nHz) -- MAIN data set from 2026-11-14.
LIKELIHOOD (ceffyl convention, Ceffyl.ln_likelihood):
  h_c = (H0/h) sqrt(3 h^2 Omega / 2) / (pi f),  S(f) = h_c^2 / (12 pi^2 f^3 T),  T = 1/f_1,
  log10 rho = 0.5 log10 S, lnL = sum_i logpdf_i(log10 rho_i)   (values below the grid -> lower boundary,
  i.e. 'no signal' in that bin).  Only the DW spectrum is fitted (no SMBHB component).
"""
from paths import PBH_ROOT, PBHBOUNDS, CEFFYL, NG15, NPROC
import numpy as np

CEF = CEFFYL + "/data/"
H0_over_h = 3.2407792894e-18     # Hz
SETS = {"EPTA": CEF + "epta_dr2_new_hd", "PPTA": CEF + "ppta_dr3_hd",
        # NANOGrav 15-yr KDE free spectra (Zenodo 10.5281/zenodo.10344086, v1.0.0, supplied by the user 2026-11-14)
        "NG15": NG15 + "/30f_fs{hd}_ceffyl", "NG15cp": NG15 + "/30f_fs{cp}_ceffyl"}
NFREQ = {"EPTA": 9, "PPTA": 10, "NG15": 14, "NG15cp": 14}

class FreeSpec:
    def __init__(self, name, nfreq=None):
        d = SETS[name]
        if nfreq is None:
            nfreq = NFREQ.get(name)
        self.f = np.load(d + "/freqs.npy")
        self.grid = np.load(d + "/log10rhogrid.npy")
        self.dens = np.load(d + "/density.npy")[0]
        if nfreq is not None:
            self.f, self.dens = self.f[:nfreq], self.dens[:nfreq]
        self.T = 1 / np.load(d + "/freqs.npy")[0]
        db = self.grid[1] - self.grid[0]
        self.edges = self.grid - 0.5 * db

    def lnL(self, h2Omega):
        """h2Omega: (n, nf) evaluated at self.f -> lnL (n,)"""
        hc = H0_over_h * np.sqrt(1.5 * np.maximum(h2Omega, 1e-300)) / (np.pi * self.f)
        S = hc**2 / (12 * np.pi**2 * self.f**3 * self.T)
        lr = 0.5 * np.log10(S)
        idx = np.clip(np.searchsorted(self.edges, lr) - 1, 0, len(self.grid) - 1)
        return self.dens[np.arange(len(self.f))[None, :], idx].sum(axis=1)

    def lnL0(self):
        return self.lnL(np.zeros((1, len(self.f))))[0]

def h2Omega_powerlaw(f, log10A, gamma=13 / 3):
    fyr = 1 / 3.15576e7
    hc = 10**np.asarray(log10A)[..., None] * (f / fyr)**((3 - gamma) / 2)
    return 2 * np.pi**2 / (3 * H0_over_h**2) * f**2 * hc**2

def profile_smbhb(F, h2O_extra, lA=np.linspace(-17, -13.5, 141), gamma=13 / 3):
    """max over SMBHB amplitude of lnL(SMBHB + extra);  h2O_extra: (n, nf)."""
    pl = h2Omega_powerlaw(F.f, lA, gamma)                      # (nA, nf)
    best = np.full(len(h2O_extra), -np.inf)
    for k in range(len(lA)):
        best = np.maximum(best, F.lnL(h2O_extra + pl[k][None, :]))
    return best
