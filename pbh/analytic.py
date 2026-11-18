r"""
Analytic (closed-form or one-line quadrature) estimates for every mechanism and every figure, compared with the
numerical results of the store.  Output: analytic.json (printed summary).  Used for Appendix "Analytic estimates".

=============================== ASSUMPTIONS ===============================
* Estimates use leading-order formulas; prior moments of the parameters are taken from the stored prior samples
  (they define the Gaussian, not the pushforward).  g_* is held constant inside each estimate unless stated.
* 'Numerical' always refers to the stored results (results/<model>/<cond>, bgscan, robustness.json).
* Gaussian pushforward model (GPM) for the maps: the reaching samples of a condition are replaced by a bivariate
  Gaussian in (log10 M, log10 f) with their weighted mean m and covariance C and total probability P; then
  p_c = P N(x_c; m, s^2 C) (1/3)^2 and a cell is reached if p_c >= eps.  Its reached area is an ellipse,
  A = 2 pi s^2 sqrt|C| ln(P a / (2 pi s^2 sqrt|C| eps)),  a = (1/3)^2  (zero if the logarithm is negative).
===========================================================================
"""
import json, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.argv = sys.argv[:1]
import results_store as rs
import fig_combined as fc
import curvaton_pipeline as cp
from scipy.special import erfc, ndtr
from scipy.stats import norm

OUT = {}
MP, MSUN_G = cp.MPl, 1.98847e33
AS = 2.1e-9
LN10 = np.log(10)
def put(key, analytic, numeric, note=""):
    OUT[key] = dict(analytic=analytic, numeric=numeric, note=note)
    print(f"{key:44s} analytic {analytic}   numeric {numeric}   {note}")

# =============================================================== A. backgrounds
import background as bgm
b0 = bgm.solve("E", 1.0, 0.0, cp.BG["Treh"])
N0 = b0["Nst"]
put("A1 Starobinsky n_s (1-2/N)", round(1 - 2 / N0, 5), round(b0["ns"], 5))
put("A1 Starobinsky r (12/N^2)", round(12 / N0**2, 5), round(b0["r"], 5))
# equivalent N-shift of the exact slow-roll n_s:  1 - 2/(N + dN) = n_s(exact)
dN_shift = 2 / (1 - b0["ns"]) - N0
put("A1 N-shift of exact n_s", round(dN_shift, 2), None, "n_s(exact) = 1 - 2/(N + shift)")

def N_analytic(alpha, w, Treh, g=106.75):
    """N_* from k_* = a_* H_*:  closed form with H_* ~ const, rho_end = 0.43 V_*, H_*/H_end = 1.5."""
    N = 55.0
    for _ in range(30):
        r = 12 * alpha / N**2
        V = 1.5 * np.pi**2 * AS * r * MP**4
        rho_end = 0.43 * V
        rho_reh = np.pi**2 / 30 * g * Treh**4
        Hreh = np.sqrt(rho_reh / 3) / MP
        lnaH_reh = np.log((3.931 / g)**(1 / 3) * cp.T0 / Treh * Hreh * cp.GeV_to_Mpcinv)
        N = lnaH_reh + (0.5 - 1 / (3 * (1 + w))) * np.log(rho_end / rho_reh) - np.log(cp.kstar) + np.log(1.5)
    return N
pr = np.load("bg_prior.npz")
X, R = pr["X_E"], pr["R_E"]
ok = np.isfinite(R[:, 2])
Na = np.array([N_analytic(10**a, w, 10**t) for a, w, t in X[ok]])
put("A2 N_* closed form vs exact (rms, max)", [round(float(np.sqrt(np.mean((Na - R[ok, 2])**2))), 2),
    round(float(np.max(abs(Na - R[ok, 2]))), 2)], "exact slow-roll solution (6000 box points, E-model)")
Nmin, Nmax = N_analytic(1.0, 0.0, 5e-3), N_analytic(1.0, 0.5, 5e-3)
put("A3 achievable N_* range (alpha=1)", [round(Nmin, 1), round(Nmax, 1)],
    [round(float(np.nanmin(R[:, 2])), 1), round(float(np.nanmax(R[:, 2])), 1)])
NP_lo = 2 / (1 - (0.9649 - 2 * 0.0042)) - dN_shift
NA_lo = 2 / (1 - (0.9743 - 2 * 0.0034)) - dN_shift
okP = pr["ok_E_planck"]
put("A3 Planck lower N_* (with shift)", round(NP_lo, 1), round(float(R[okP, 2].min()), 1))
put("A4 ACT requires N_* >= (with shift) vs achievable max", [round(NA_lo, 1), round(Nmax, 1)],
    "allowed region empty (9/6000 E, 0/6000 T)")
put("A5 alpha_max from r<0.036 at N_max (12 alpha/N^2)", round(float(np.log10(0.036 * Nmax**2 / 12)), 2),
    round(float(X[okP, 0].max()), 2), "log10 alpha")

# =============================================================== B. quadratic curvaton
BG = cp.BG
Hst, Treh = BG["Hst"], BG["Treh"]
def M_of_T(T):            # gamma M_H, solar masses (cp tables)
    return np.interp(np.log(T), np.log(cp._Ttab), cp._M_T)
def lnk_of_T(T):
    return np.log(cp.aH_Mpc(T))
Tgrid = np.logspace(-2, np.log10(Treh) - 0.5, 400)
PI = 4 / 9 * Hst**2 / (24 * np.pi**2 * MP**2) * Treh / Tgrid
dNk = lnk_of_T(Tgrid) - np.log(cp.kstar)
PII = 0.01 * AS * ((4 / 9) / (0.01 * AS))**(dNk / BG["Nst"])
Pmax = np.minimum(PI, PII); j = np.argmax(Pmax)
# numerical: max over the allowed prior samples of P_zeta at the decay scale
d = np.load("samples.npz"); th = d["th"]
best = 0.0; bestM = None
for i in range(0, len(th), 20000):
    t = th[i:i + 20000]
    okc, aux = cp.constraints(t)
    t, Px, Om = t[okc], aux["Px"][okc], aux["Om"][okc]
    if not len(t):
        continue
    rr = 3 * Om / (4 - Om)
    lnk_dec = np.log(cp.aH_Mpc(10**t[:, 2]))
    idx = np.array([np.argmin(abs(cp.lnk_of_N - l)) for l in lnk_dec])
    Pz = (2 * rr / 3)**2 * Px[np.arange(len(t)), idx]
    k = np.argmax(Pz)
    if Pz[k] > best:
        best, bestM = float(Pz[k]), float(np.log10(M_of_T(10**t[k, 2])))
put("B1 quadratic curvaton max P_zeta(k_dec)", [float(f"{Pmax[j]:.2g}"), round(float(np.log10(M_of_T(Tgrid[j]))), 1)],
    [float(f"{best:.2g}"), None if bestM is None else round(bestM, 1)], "[P_zeta, log10 M at the maximum]")
xc = np.exp(1.5) - 1                                  # zeta_c = 1, Omega_dec -> 1 (SVW): (1+x)^2 = e^3
log10beta = -xc**2 / (2 * 9 / 4 * Pmax[j] * LN10)
lfmax_num = float(np.nanmax(d["lf_1.0"][np.isfinite(d["lf_1.0"])])) if np.isfinite(d["lf_1.0"]).any() else None
put("B2 quadratic curvaton log10 f_max", float(f"{log10beta:.3g}"), lfmax_num,
    "Gaussian tail with x_c = e^{3/2}-1; numeric: max over prior samples (-inf if none finite)")

# =============================================================== C. inflating curvaton
NPlow = 2 / (1 - 0.9565)
put("C1 inflating curvaton N_2 max (leading / shifted)", [round(N0 - NPlow, 1), round(N0 - NPlow + dN_shift, 1)], 6.8)

# =============================================================== D. axion-like curvaton
a = rs.load("axion_curvaton", "opt"); P, _ = rs.maps(a)
ncell = int((P > 0).sum())
put("D1 axion: P_total / N_support vs max p_c", float(f"{P.sum() / max(ncell, 1):.2g}"), float(f"{P.max():.2g}"),
    f"support {ncell} cells; eps = 1e-5")

# =============================================================== E. FOPT (non-covariant)
def lnPcoll(b, dc):
    return -1.024 * b**0.6921 * (1 + dc)**(0.8831 * b)
def log10f(lT, b, dc):
    return lnPcoll(b, dc) / LN10 - np.log10(2.2e-8) + lT - np.log10(0.140)
g = rs.load("fopt", "GV_dc0.5"); S = g["theta"]
mu, Cv = S.mean(0), np.cov(S.T)
# band probability: for each log T, the b-interval with -12 < log f < 0 under the conditional Gaussian
lTs = np.linspace(S[:, 0].min(), S[:, 0].max(), 400)
bgrid = np.linspace(3, 30, 20001)
Pb = 0.0
for lT in lTs:
    m_b = mu[1] + Cv[0, 1] / Cv[0, 0] * (lT - mu[0]); s_b = np.sqrt(Cv[1, 1] - Cv[0, 1]**2 / Cv[0, 0])
    lf = log10f(lT, bgrid, 0.5)
    inb = (lf > -12) & (lf < 0)
    if inb.any():
        b1, b2 = bgrid[inb].min(), bgrid[inb].max()
        Pb += (ndtr((b2 - m_b) / s_b) - ndtr((b1 - m_b) / s_b)) * norm.pdf(lT, mu[0], np.sqrt(Cv[0, 0])) * (lTs[1] - lTs[0])
lfn = g["lf"]
put("E1 FOPT P(1e-12 < f < 1), delta_c = 0.5", round(float(Pb), 4), round(float(np.mean((lfn > -12) & (lfn < 0))), 4),
    "Gaussian prior moments of (log T, beta/H); bimodality")
def band_prob(mu, Cv, dc=0.5):
    out = 0.0
    for lT in lTs:
        m_b = mu[1] + Cv[0, 1] / Cv[0, 0] * (lT - mu[0]); s_b = np.sqrt(Cv[1, 1] - Cv[0, 1]**2 / Cv[0, 0])
        lf = log10f(lT, bgrid, dc); inb = (lf > -12) & (lf < 0)
        if inb.any():
            b1, b2 = bgrid[inb].min(), bgrid[inb].max()
            out += (ndtr((b2 - m_b) / s_b) - ndtr((b1 - m_b) / s_b)) * norm.pdf(lT, mu[0], np.sqrt(Cv[0, 0])) * (lTs[1] - lTs[0])
    return out
for s, tag in ((0.5, "ps0.5"), (2.0, "ps2")):
    z = np.load(f"bgscan/robust/{tag}.npz")["fopt__GV_dc0.5"]
    put(f"E1b FOPT P(1e-12<f<1), prior width x{s}", round(float(band_prob(mu, Cv * s**2)), 4),
        round(float(np.mean((z[1] > -12) & (z[1] < 0))), 4), "same formula with covariance x s^2")
bstar = bgrid[np.argmin(abs(lnPcoll(bgrid, 0.5) / LN10 - (-6 - 7.66)))]
def A_env(b, T):
    gs = cp.gstar_s(T)
    return 1.67e-5 / b**2 * (100 / gs)**(1 / 3) * (0.11 / 1.42) / 0.674**2
def fpeak(b, T):
    return 16.5e-6 * (0.62 / 2.7) * b * (T / 100) * (cp.gstar_s(T) / 100)**(1 / 6)
Ts = np.logspace(-2.5, 9, 4000)
# LVK: above-peak tail S ~ 1.357/x at f = 20 Hz; allowed if Omega(20 Hz) < 5.8e-9 (f_p < 20 Hz branch)
OmL = A_env(bstar, Ts) * 1.357 * fpeak(bstar, Ts) / 20.0
Tlvk = Ts[np.argmax(OmL > 5.8e-9)]
Mlight = np.log10(0.5 * np.log10(20 / cp.gstar(Tlvk)) * 0 + (20 / cp.gstar(Tlvk))**0.5 * (0.14 / Tlvk)**2)
# PTA (pointwise): below-peak tail S ~ 3.8 x^2.8 at f_hi = 3e-8 Hz must stay below the signal 1.51e-8 (f/f_yr)^{2/3}
fyr = 1 / 3.15576e7; fhi = 3e-8
OmP = A_env(bstar, Ts) * 3.8 * (fhi / fpeak(bstar, Ts))**2.8
Tpta = Ts[np.argmax(OmP < 1.51e-8 * (fhi / fyr)**(2 / 3))]
Mheavy = np.log10((20 / cp.gstar(Tpta))**0.5 * (0.14 / Tpta)**2)
Pm, _ = rs.maps(g); xs = rs.xc[(Pm >= rs.EPS).any(1)]
put("E2 FOPT beta/H at the band centre", round(float(bstar), 2), None)
put("E3 FOPT light edge (LVK), log10 M", round(float(Mlight), 1), round(float(xs.min()), 1), "numeric: lightest reached cell")
put("E4 FOPT heavy edge (PTA pointwise), log10 M", round(float(Mheavy), 1), [-0.6, round(float(xs.max()), 1)],
    "numeric: [pointwise run (2026-11-13), NG15 likelihood: heaviest reached cell]")
put("E5 FOPT minimum T_reh (= T at the PTA edge), log10 GeV", round(float(np.log10(Tpta)), 2), -0.9,
    "numeric: background family, reach vanishes below")

# =============================================================== F. domain walls
def t_of_T(T):
    return 1 / (2 * cp.H_rad(T))
def sigma13(T, al):
    return (3 * al * MP**2 / (4 * 0.8 * t_of_T(T)))**(1 / 3)
def M_dw(T, al):
    return 4 * np.pi * MP**2 * t_of_T(T) / np.sqrt(3 * al) * cp.GeV_to_g / MSUN_G
put("F1 DW T_reh threshold sigma^{1/3} at NG15 best fit, log10 GeV", round(float(np.log10(sigma13(0.12, 0.16))), 2), 5.2)
put("F2 DW mass at NG15 best fit (monochromatic), log10 Msun", round(float(np.log10(M_dw(0.12, 0.16))), 2),
    round(float(np.median(rs.load("dw", "ptaL_Lt0.8_mono")["lM"][(rs.load("dw", "ptaL_Lt0.8_mono")["lf"] > -12)])), 2))
# mass range of the PTA band T_ann = 60-180 MeV, alpha = 0.06-0.32
put("F3 DW mass range over the NG15 band, log10 Msun",
    [round(float(np.log10(M_dw(0.18, 0.32))), 2), round(float(np.log10(M_dw(0.06, 0.06))), 2)], [-0.1, 1.4],
    "numeric: 5-95% of reaching samples over L/t (width)")
# light edge of the reach, case (i): sigma^{1/3} = T_reh with alpha_ann ~ 0.1-1
Tl = np.logspace(-2, 9, 4000)
for al in (0.1, 1.0):
    Tedge = Tl[np.argmax(sigma13(Tl, al) > Treh)]
    OUT.setdefault("F4_list", []).append((al, float(np.log10(M_dw(Tedge, al)))))
dwu = rs.load("dw", "ulL_Lt0.8_width"); Pu, _ = rs.maps(dwu)
put("F4 DW light edge from sigma^{1/3} < T_reh (alpha 0.1, 1), log10 M", [round(x[1], 1) for x in OUT["F4_list"]],
    round(float(rs.xc[(Pu >= rs.EPS).any(1)].min()), 1), "numeric: lightest reached cell, case (i), L/t = 0.8")
put("F5 DW heavy end of the NG15 band (T=60 MeV, alpha=0.06), log10 M", round(float(np.log10(M_dw(0.06, 0.06))), 2),
    [1.2, 1.5], "numeric: robust-(A) boundary in scenarios (d),(c)")

# =============================================================== G. Fermi balls: linear-Gaussian pushforward
f = rs.load("fermiball", "base"); S = f["theta"]
mu, Cv = S.mean(0), np.cov(S.T)
gref = 10.0
cM = np.log10(1.4e21 * 0.6**3 / 1e-3 * (100 / gref)**0.25 * 100**2 * 100**3 / MSUN_G)
aM = np.array([-2.0, -3.0, 0.25, 1.0]); cf = np.log10(1.82e10 * (gref / 100)**0.25)
af = np.array([1.0, 0.0, 0.25, 1.0])
mM, sM = cM + aM @ mu, np.sqrt(aM @ Cv @ aM)
mF, sF = cf + af @ mu, np.sqrt(af @ Cv @ af)
rho = (aM @ Cv @ af) / (sM * sF)
lMall, lfe = f["lM_all"], f["lf_eff"]
Mev = np.log10(5.1e14 / MSUN_G)
put("G1 Fermi-ball mean / sd of log10 M", [round(mM, 2), round(sM, 2)], [round(float(lMall.mean()), 2), round(float(lMall.std()), 2)])
put("G2 Fermi-ball mean / sd of log10 f", [round(mF, 2), round(sF, 2)], [round(float(lfe.mean()), 2), round(float(lfe.std()), 2)],
    "f_eff (would-be abundance, incl. evaporated)")
put("G3 Fermi-ball corr(log M, log f)", round(float(rho), 2), round(float(np.corrcoef(lMall, lfe)[0, 1]), 2))
put("G4 Fermi-ball evaporated fraction", round(float(norm.cdf((Mev - mM) / sM)), 3), round(float(np.mean(lMall < Mev)), 3))

# =============================================================== H. walls from inflation
Mev_sun = 5.1e14 / MSUN_G
put("H1 walls: peak mass on the reference background, log10 M", round(float(np.log10(Mev_sun)), 2), -18.59)
T_in = np.sqrt(4.8e-2 / 1e-18)
put("H2 walls: T_reh below which the abundance peak enters the window, log10 GeV", round(float(np.log10(T_in)), 2), 8.0,
    "numeric: reach rises below ~1e8 GeV (Fig. 4)")
# median peak mass in the background mixture: prior on log T_reh truncated to the in-window range
m_, s_ = float(pr["mean_E_planck"][2]), float(np.sqrt(pr["cov_E_planck"][2, 2]))
lo_, hi_ = np.log10(5e-3), np.log10(T_in)
qmed = norm.ppf(0.5 * (norm.cdf((lo_ - m_) / s_) + norm.cdf((hi_ - m_) / s_))) * s_ + m_
put("H3 walls: median peak mass (background mixture), log10 M", round(float(np.log10(4.8e-2) - 2 * qmed), 1), -11.6,
    "analytic: median of M_H(T_reh) over the T_reh prior restricted to the window")

# =============================================================== I. cosmic strings
cs_max = 6.3e-11; OmDM = 0.120 / 0.674**2
put("I1 strings: f_tot max = c_s,max/Omega_DM (M_evap/M_*)^{-1/2}", float(f"{cs_max / OmDM * (5.1 / 5)**-0.5:.2g}"), 2.5e-10)
gmu_min = 1.5e-11 * (OmDM * 1e-12 / 300)**(2 / 3)
eta_min = 1.2209e19 * np.sqrt(gmu_min / (2 * np.pi))
Tmax_ref = 1.6e12
Treh_min = cp.BG["Treh"] * (eta_min / Tmax_ref)**2
put("I2 strings: T_reh threshold for f_tot > 1e-12 (T_max ~ T_reh^{1/2}), log10 GeV", round(float(np.log10(Treh_min)), 1),
    [1.97, 3.97], "numeric: [lowest T_reh with reach, highest T_reh with zero reach]")

# =============================================================== J. USR feature
nodes = np.load("usr_nodes.npy", allow_pickle=True)
rat, dnum, KP = [], [], []
for n in nodes:
    R_ = n["recs"]
    dd = np.array([q["delta"] for q in R_]); fG = np.array([q["f_G"] for q in R_]); PP = np.array([q["Pmax"] for q in R_])
    o = np.argsort(dd); dd, fG, PP = dd[o], fG[o], PP[o]
    m = (fG > 0) & np.isfinite(fG)
    if m.sum() < 4:
        continue
    lf = np.log10(fG[m]); ld = np.log10(dd[m])
    if lf.max() < 0 or lf.min() > -12:
        continue
    ordr = np.argsort(lf)
    d1 = 10**np.interp(0.0, lf[ordr], ld[ordr]); d12 = 10**np.interp(-12.0, lf[ordr], ld[ordr])
    lnb0 = np.log(1.0 / 5e14)      # beta at f = 1 for M ~ 1e-13 Msun (f/beta ~ 5e14)
    rat.append(d1 * (np.sqrt(1 + 12 * LN10 / abs(lnb0)) - 1)); dnum.append(d12 - d1)
    KP.append(np.median(PP * dd**2))
put("J1 USR shell width Delta ln A (median over nodes)", float(f"{np.median(rat):.2g}"), float(f"{np.median(dnum):.2g}"),
    "analytic: ln beta ~ -delta^2 with delta(f=1) from the node; numeric: delta(f=1e-12) - delta(f=1)")
put("J2 USR P_max delta^2 (median)", float(f"{np.median(KP):.2g}"), None, "P_max ~ K delta^-2")
logAc = np.median([np.log10(n["Ac"]) for n in nodes])
sdA = (logAc + 6) / np.sqrt(12); mA = -4.32
Pshell = 0.469 * norm.pdf(logAc, mA, sdA) * np.median(dnum) / LN10
put("J3 USR P(1e-12<f<1)", float(f"{Pshell:.2g}"), float(f"{6.05e-6 - 3.73e-6:.2g}"),
    "allowed fraction x prior density at log A_c x shell width")

# =============================================================== K. figures: Gaussian pushforward model
def gpm(d, eps, s=1.0):
    w = d["w"] if "w" in d else np.full(len(d["lM"]), 1.0 / len(d["lM"]))
    sel = (d["ok"] > 0) & (d["lf"] > -12) & (d["lM"] >= rs.xe[0]) & np.isfinite(d["lM"])
    if w[sel].sum() <= 0:
        return np.zeros((len(rs.xc), len(rs.yc)), bool)
    P = w[sel].sum(); x = np.vstack([d["lM"][sel], d["lf"][sel]])
    m = np.average(x, axis=1, weights=w[sel]); C = np.cov(x, aweights=w[sel]) * s**2
    Ci = np.linalg.inv(C); dt = np.sqrt(np.linalg.det(C))
    XX, YY = np.meshgrid(rs.xc, rs.yc, indexing="ij")
    D = np.stack([XX - m[0], YY - m[1]], -1)
    q = np.einsum("...i,ij,...j->...", D, Ci, D)
    pc = P * np.exp(-0.5 * q) / (2 * np.pi * dt) / 9.0
    return pc >= eps
cache = {}
def load(k):
    if k not in cache:
        cache[k] = rs.load(*k)
    return cache[k]
allowed = ~fc.excluded
fig_rows = {}
for eps in (1e-6, 1e-5, 1e-4):
    for sc, s_ in fc.SCEN.items():
        reachA = np.zeros_like(allowed); reachN = np.zeros_like(allowed)
        for m, ks in s_.items():
            for k in ks:
                d = load(k)
                reachA |= gpm(d, eps)
                reachN |= rs.maps(d)[0] >= eps
        fA = float(((~reachA) & allowed).sum() / allowed.sum()); fN = float(((~reachN) & allowed).sum() / allowed.sum())
        fig_rows[f"{sc[0]}_{sc[1]}_eps{eps:.0e}"] = (round(fA, 3), round(fN, 3))
for k, v in fig_rows.items():
    put(f"K1 (A) fraction GPM vs numeric {k}", v[0], v[1])
# prior width: GPM with C -> s^2 C at fixed P, compared with the reruns (robustness.json)
rob = json.load(open("robustness.json"))
for s, tag in ((0.5, "ps0.5"), (2.0, "ps2")):
    for sc, s_ in fc.SCEN.items():
        reachA = np.zeros_like(allowed)
        for m, ks in s_.items():
            for k in ks:
                reachA |= gpm(load(k), 1e-5, s)
        fA = float(((~reachA) & allowed).sum() / allowed.sum())
        put(f"K2 (A) fraction GPM vs rerun, width x{s} {sc[0]}_{sc[1]}", round(fA, 3), round(rob[tag][f"{sc[0]}_{sc[1]}"]["A"]["val"], 3))
# per-mechanism ellipse areas (cells) vs numeric reached cells
for k in [("fopt", "GV_dc0.5"), ("dw", "ulL_Lt0.8_width"), ("fermiball", "base"), ("dw", "ptaL_Lt0.8_width")]:
    d = load(k)
    row = []
    for eps in (1e-6, 1e-5, 1e-4):
        row.append((int((gpm(d, eps) & allowed).sum()), int(((rs.maps(d)[0] >= eps) & allowed).sum())))
    put(f"K3 reached cells GPM vs numeric {k[0]}/{k[1]} eps=1e-6,-5,-4", [r[0] for r in row], [r[1] for r in row])

json.dump(OUT, open("analytic.json", "w"), indent=1, default=float)
print("wrote analytic.json")
