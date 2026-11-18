r"""
Variable inflationary background (2026-11-15, HANDOFF §8 item 1): alpha-attractor plateau family + free reheating.
Replaces the fixed Starobinsky background of curvaton_pipeline.py when the environment variable PBH_BG points to a
JSON file written by write_bg() (curvaton_pipeline reads it at import; every model module then sees the new BG).

=============================== ASSUMPTIONS ===============================
Potentials (Kallosh & Linde, arXiv:1306.5220; Kallosh, Linde & Roest, arXiv:1311.0472), M_P = 1:
    E-model  V = V0 (1 - exp(-sqrt(2/(3 alpha)) phi))^{2n}     (alpha = 1, n = 1 is Starobinsky)
    T-model  V = V0 tanh^{2n}(phi / sqrt(6 alpha))
  Leading order for both: n_s = 1 - 2/N_*, r = 12 alpha / N_*^2.
Slow roll throughout inflation (as in curvaton_pipeline.py): H^2 = V/3, eps_V = (V'/V)^2/2, eta_V = V''/V,
  n_s = 1 - 6 eps + 2 eta, r = 16 eps, alpha_s = -dn_s/dN (numerical); end of inflation at eps_V = 1;
  rho_end = (3/2) V_end.  A_s = 2.1e-9 at k_* = 0.05 Mpc^-1 fixes V0 = 24 pi^2 eps_* A_s / v_*.
Reheating (free, replaces scalaron -> Higgs): parameters T_reh and w_reh.
  Oscillations about the minimum V ~ phi^{2n} have <w> = (n-1)/(n+1) (Turner 1983) -> n = (1+w)/(1-w);
  the potential exponent n and w_reh are therefore ONE parameter.  Constant w from the end of inflation until
  rho = rho_reh = (pi^2/30) g_rho(T_reh) T_reh^4 (instant conversion), then radiation.
  ln(a_end H_end) = ln(a_reh H_reh) + [1/2 - 1/(3(1+w))] ln(rho_end/rho_reh).
  Gamma := H_reh = sqrt(rho_reh/3) (decay rate defined by Gamma = H at reheating; same convention as before).
  Requires rho_reh <= rho_end.  Fragmentation / preheating of phi^{2n} condensates is ignored (要確認 for n > 1).
T_max: maximal radiation temperature during reheating from
  d ln rho_phi/dN = -3(1+w) - Gamma/H,  d ln rho_r/dN = -4 + (Gamma/H) rho_phi/rho_r  (GKR hep-ph/0005123 with
  constant Gamma; for n > 1 the effective rate is field-dependent -- 要確認), instant thermalisation.
CMB conditions on the background (define the allowed region of psi = (log10 alpha, w, log10 T_reh)):
  Planck band: n_s = 0.9649 +- 0.0042 (Planck 2018 X, arXiv:1807.06211), 2 sigma.
  ACT band   : n_s = 0.9743 +- 0.0034 (ACT DR6 P-ACT-LB, arXiv:2503.14452), 2 sigma.
  r < 0.036 (BICEP/Keck 2021, arXiv:2110.00483).  T_reh > 5 MeV (BBN, as elsewhere in the project).
Box (theory choice, flagged): log10 alpha in [-2, 2], w in [0, 1/2] (n = 1 ... 3), log10 T_reh/GeV in
  [log10 5e-3, log10 T_reh,max(rho_reh = rho_end)].  E- and T-models are two discrete options, weight 1/2 each.
===========================================================================
"""
import json, os
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
import curvaton_pipeline as cp          # constants and the radiation-era tables (aH_Mpc, g_*)

MP = cp.MPl
AS, KSTAR = 2.1e-9, 0.05
NS = {"planck": (0.9649, 0.0042), "act": (0.9743, 0.0034)}
R_MAX = 0.036
LT_MIN = np.log10(5e-3)
BOX = {"la": (-2.0, 2.0), "w": (0.0, 0.5)}

def _eps_eta(model, phi, alpha, n):
    if model == "E":
        c = np.sqrt(2 / (3 * alpha)); u = np.exp(-c * phi)
        d1 = 2 * n * c * u / (1 - u)
        eta = 2 * n * c**2 * u * (2 * n * u - 1) / (1 - u)**2
        v = (1 - u)**(2 * n)
    else:
        b = np.sqrt(6 * alpha); t = np.tanh(phi / b)
        d1 = 2 * n * (1 - t**2) / (b * t)
        eta = d1**2 - 2 * n * (1 + t**2) * (1 - t**2) / (b**2 * t**2)
        v = t**(2 * n)
    return 0.5 * d1**2, eta, v

class Plateau:
    """slow-roll tables for one (model, alpha, n): N(phi), eps, eta, v on a fine phi grid."""
    def __init__(self, model, alpha, n, Nmax=130.0):
        self.model, self.alpha, self.n = model, alpha, n
        scale = np.sqrt(alpha)
        # end of inflation: eps_V = 1 (bisection on a log grid)
        ph = np.logspace(-4, 2.5, 20000) * scale
        e, _, _ = _eps_eta(model, ph, alpha, n)
        i = np.argmax(e < 1.0)
        lo, hi = ph[i - 1], ph[i]
        for _ in range(80):
            m = 0.5 * (lo + hi)
            if _eps_eta(model, m, alpha, n)[0] > 1: lo = m
            else: hi = m
        self.phi_end = hi
        phi = self.phi_end + np.concatenate([[0], np.logspace(-6, 3, 60000)]) * scale
        e, eta, v = _eps_eta(model, phi, alpha, n)
        N = cumulative_trapezoid(1 / np.sqrt(2 * e), phi, initial=0.0)
        k = np.argmax(N > Nmax) + 1
        self.phi, self.eps, self.eta, self.v, self.N = phi[:k], e[:k], eta[:k], v[:k], N[:k]
        self.ns_N = 1 - 6 * self.eps + 2 * self.eta
    def at(self, N):
        return {q: np.interp(N, self.N, getattr(self, q)) for q in ("eps", "eta", "v", "ns_N")}

def T_max(Hend, Gam, w):
    def rhs(x, y):
        rp, rr = np.exp(y)
        H = np.sqrt((rp + rr) / 3) / MP
        return [-3 * (1 + w) - Gam / H, -4 + Gam / H * rp / rr]
    y0 = [np.log(3 * MP**2 * Hend**2), np.log(3 * MP**2 * Hend**2 * 1e-30)]
    xmax = 2 + np.log(Hend / Gam) / (1.5 * (1 + w)) * 1.3 + 5
    sol = solve_ivp(rhs, [0, xmax], y0, rtol=1e-8, atol=1e-10, dense_output=True)
    x = np.linspace(0, xmax, 20001)
    rr = np.exp(sol.sol(x)[1])
    T = (30 * rr / (np.pi**2 * 106.75))**0.25
    for _ in range(3):
        T = (30 * rr / (np.pi**2 * cp.gstar(T)))**0.25
    i = np.argmax(T)
    return float(T[i])

def solve(model, alpha, w, Treh, P=None, full=False):
    """background for psi = (model, alpha, w, T_reh).  returns dict (BG-compatible keys + CMB observables)."""
    n = (1 + w) / (1 - w)
    P = P or Plateau(model, alpha, n)
    rho_reh = np.pi**2 / 30 * cp.gstar(Treh) * Treh**4
    Nst = 55.0
    for _ in range(60):
        q = P.at(Nst)
        V0 = 24 * np.pi**2 * q["eps"] * AS / q["v"]          # M_P = 1 units
        Vend = V0 * P.v[0]
        rho_end = 1.5 * Vend * MP**4
        Hst = np.sqrt(V0 * q["v"] / 3) * MP
        Hend = np.sqrt(rho_end / 3) / MP
        if rho_reh > rho_end:
            return None
        lnaHend = np.log(cp.aH_Mpc(Treh)) + (0.5 - 1 / (3 * (1 + w))) * np.log(rho_end / rho_reh)
        Nnew = lnaHend - np.log(KSTAR) + np.log(Hst / Hend)
        if abs(Nnew - Nst) < 1e-7:
            break
        Nst = Nnew
    q = P.at(Nst)
    ns = q["ns_N"]
    als = -float(np.interp(Nst, P.N, np.gradient(P.ns_N, P.N)))
    out = dict(model=model, alpha=alpha, w=w, n=n, Treh=Treh, Nst=Nst, ns=ns, r=16 * q["eps"], alpha_s=als,
               V0=V0, mu=np.sqrt(V0) * MP, Hst=Hst, Hend=Hend, Gam=np.sqrt(rho_reh / 3) / MP,
               aHend=float(np.exp(lnaHend)), Tend_inst=(rho_end * 30 / (np.pi**2 * 106.75))**0.25)
    if full:
        Ngrid = np.linspace(0.0, Nst, 1200)
        Hgrid = np.sqrt(V0 * np.interp(Ngrid, P.N, P.v) / 3) * MP
        out.update(Ngrid=Ngrid.tolist(), Hgrid=Hgrid.tolist(), Tmax=T_max(Hend, out["Gam"], w))
    return out

def lT_max(model, alpha, w, P=None):
    """log10 T_reh at which rho_reh = rho_end (instant reheating)."""
    P = P or Plateau(model, alpha, (1 + w) / (1 - w))
    lo, hi = LT_MIN, 17.0
    for _ in range(50):
        m = 0.5 * (lo + hi)
        if solve(model, alpha, w, 10**m, P) is None: hi = m
        else: lo = m
    return lo

def cmb_ok(b, band):
    mu, sd = NS[band]
    return (b is not None) and (abs(b["ns"] - mu) <= 2 * sd) and (b["r"] < R_MAX)

def write_bg(b, path):
    d = {k: (float(v) if isinstance(v, (np.floating, float, int)) and not isinstance(v, bool) else v) for k, v in b.items()}
    json.dump(d, open(path, "w"), indent=1)

if __name__ == "__main__":
    import time
    t = time.time()
    b = solve("E", 1.0, 0.0, 2.5e9, full=True)
    print("E-model alpha=1, w=0, T_reh=2.5e9 (cf. Starobinsky fixed BG: N_*=51.0, n_s=0.9623, H_*=1.58e13, H_end=1.05e13):")
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in b.items() if k not in ("Ngrid", "Hgrid")})
    print("fixed BG:", {k: f"{v:.4g}" for k, v in cp.BG.items()}, f" t={time.time()-t:.2f}s")
