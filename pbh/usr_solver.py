r"""
Inflaton-generated small-scale power: exact background + Mukhanov-Sasaki (MS) solver for single-field
potentials with a localized feature (ultra-slow-roll, USR).

=============================== ASSUMPTIONS ===============================
Units: reduced Planck mass M_P = 1 (code); physical k in Mpc^-1.
Potential (Mishra & Sahni, JCAP 04 (2020) 007, arXiv:1911.00057: base potential times a Gaussian bump):
    V(phi) = V0 * v_b(phi) * [1 + A exp(-(phi - phi0)^2 / (2 s^2))]
  base 'staro':  v_b = (3/4)(1 - exp(-sqrt(2/3) phi))^2              (Starobinsky; V0 = mu^2)
  base 'Emodel': v_b = (3/4)(1 - exp(-sqrt(2/(3 alpha)) phi))^2      (alpha-attractor E-model; alpha = 1 is Starobinsky)
Background: exact equations in e-folds N (no slow roll),
    H^2 = V / (3 - pi^2/2),   dpi/dN = -(3 - pi^2/2)(pi + V_phi/V),   pi = dphi/dN;
    eps1 = pi^2/2, eps2 = dln eps1/dN.  End of inflation at eps1 = 1.
    'Trapped' (field stops before the bump top: pi changes sign) -> flagged, excluded.
MS equation in N for R_k:  R'' + (3 - eps1 + eps2) R' + (k/aH)^2 R = 0, Bunch-Davies at k/aH = KSTART,
    R = 1/(z sqrt(2k)) e^{-ik tau},  R' = -(i k/aH + 1 + eps2/2) R  (z = a sqrt(2 eps1)),
    P_R = k^3 |R|^2 / (2 pi^2) at the end of inflation.  Vectorised fixed-step RK4 over all modes; a mode is
    switched on when k/aH = KSTART.
Reheating (sets the k <-> N map; same as the fixed background of curvaton_pipeline.py for 'staro'):
    Gamma = mu^3/(192 pi M_P^2) with mu^2 = V0 (scalaron -> Higgs); matter domination until H = Gamma,
    rho_reh = 3 Gamma^2, a_end H_end = a_reh H_reh (rho_end/rho_reh)^{1/6}.   For 'Emodel' the same Gamma(V0)
    is used (assumption; T_reh is not predicted by the E-model).
Normalisation: V0 iterated so that P_R(k_* = 0.05 Mpc^-1) = A_s = 2.1e-9; n_s, alpha_s from a quadratic fit of
    ln P_R vs ln k over k_*/e^2 ... k_* e^2 (5 modes).
===========================================================================
"""
import numpy as np
from scipy.integrate import solve_ivp
import curvaton_pipeline as cp

MPl_GeV = cp.MPl
AS, KSTAR = 2.1e-9, 0.05
KSTART = 60.0

def make_potential(base, A, phi0, s, alpha=1.0):
    c = np.sqrt(2 / 3) if base == "staro" else np.sqrt(2 / (3 * alpha))
    def V(p):
        e = np.exp(-c * p)
        g = A * np.exp(-(p - phi0)**2 / (2 * s**2))
        return 0.75 * (1 - e)**2 * (1 + g)
    def dV(p):
        e = np.exp(-c * p)
        g = A * np.exp(-(p - phi0)**2 / (2 * s**2))
        vb = 0.75 * (1 - e)**2; dvb = 1.5 * (1 - e) * c * e
        return dvb * (1 + g) + vb * g * (-(p - phi0) / s**2)
    return V, dV

def background(V, dV, phi_ini, Nmax=400.0):
    """integrate from rest-ish slow-roll initial velocity at phi_ini down to eps1 = 1."""
    pi0 = -dV(phi_ini) / V(phi_ini)          # slow-roll attractor velocity
    def rhs(N, y):
        p, pi = y
        return [pi, -(3 - pi**2 / 2) * (pi + dV(p) / V(p))]
    def end(N, y): return y[1]**2 / 2 - 1.0
    end.terminal = True; end.direction = 1
    def stuck(N, y): return y[1]          # velocity changes sign -> trapped behind the bump
    stuck.terminal = True; stuck.direction = 1
    sol = solve_ivp(rhs, [0, Nmax], [phi_ini, pi0], events=[end, stuck], rtol=1e-10, atol=1e-12,
                    dense_output=True, max_step=0.05)
    trapped = len(sol.t_events[1]) > 0 or len(sol.t_events[0]) == 0
    return sol, trapped

def spectrum(base, A, phi0, s, alpha=1.0, phi_ini=None, lnk_extra=None, nk=160, dN=1e-3, V0_iter=3):
    """returns dict with lnk [Mpc^-1], P_R, N_star, n_s, alpha_s, V0, Treh, trapped, eps-history."""
    V, dV = make_potential(base, A, phi0, s, alpha)
    if phi_ini is None:
        phi_ini = 6.5 if base == "staro" else 6.5 * np.sqrt(alpha)
    sol, trapped = background(V, dV, phi_ini)
    out = dict(trapped=trapped)
    if trapped:
        return out
    Nend = sol.t[-1]
    Ng = np.arange(0, Nend, dN); Ng = np.append(Ng, Nend)
    p, pi = sol.sol(Ng)
    eps1 = pi**2 / 2
    H2 = V(p) / (3 - eps1)                        # in units of V0
    # eps2 = dln eps1/dN = 2 pi'/pi
    dpi = -(3 - eps1) * (pi + dV(p) / V(p))
    eps2 = 2 * dpi / pi
    lnaH_code = Ng + 0.5 * np.log(H2)              # a = e^N, H in sqrt(V0) units
    # k-mapping and normalisation (iterate V0 because Gamma depends on V0)
    V0 = 1e-10
    for _ in range(V0_iter + 1):
        mu = np.sqrt(V0)
        Gam = mu**3 / (192 * np.pi)                # M_P = 1
        rho_end = 3 * H2[-1] * V0
        rho_reh = 3 * Gam**2
        g = 106.75
        Treh = (30 * rho_reh / (np.pi**2 * g))**0.25 * MPl_GeV
        lnaHend_phys = np.log(cp.aH_Mpc(Treh)) + np.log(rho_end / rho_reh) / 6
        C = lnaHend_phys - lnaH_code[-1]          # ln k_phys = ln k_code + C
        lnk_N = lnaH_code + C                      # horizon-exit k as function of N
        # N_* : k_* = aH  (monotonic in the slow-roll part; take the first crossing)
        i = np.argmax(lnk_N >= np.log(KSTAR))
        Nst = np.interp(np.log(KSTAR), lnk_N[max(i - 2, 0):i + 2], Ng[max(i - 2, 0):i + 2])
        # power at pivot from MS (few modes) -> rescale V0
        lnk_piv = np.log(KSTAR) + np.linspace(-2, 2, 5)
        Ppiv = _ms(lnk_piv - C, Ng, lnaH_code, eps1, eps2, H2, dN)   # in units of V0
        V0 = AS / Ppiv[2]
    if Ng[i] < 5:
        out["trapped"] = True; out["reason"] = "pivot too close to start"; return out
    cfit = np.polyfit(lnk_piv - np.log(KSTAR), np.log(Ppiv * V0), 2)
    ns, als = 1 + cfit[1], 2 * cfit[0]
    lnk = np.linspace(np.log(KSTAR) - 1, lnk_N[-1] - 1.0, nk) if lnk_extra is None else lnk_extra
    P = _ms(lnk - C, Ng, lnaH_code, eps1, eps2, H2, dN) * V0
    r = 16 * np.interp(Nst, Ng, eps1)
    out.update(lnk=lnk, P=P, Nst=Nend - Nst, ns=ns, alpha_s=als, r=r, V0=V0, Treh=Treh, Nend=Nend,
               C=C, Ng=Ng, eps1=eps1, eps2=eps2, lnk_N=lnk_N)
    return out

def _ms(lnk_code, Ng, lnaH, eps1, eps2, H2, dN):
    """vectorised RK4 for R_k(N); returns P_R(k) / V0 at the end."""
    k = np.exp(lnk_code)
    nk = len(k)
    # start index for each mode: k/aH = KSTART
    istart = np.searchsorted(lnaH, lnk_code - np.log(KSTART))
    istart = np.clip(istart, 0, len(Ng) - 2)
    R = np.zeros(nk, complex); dR = np.zeros(nk, complex)
    on = np.zeros(nk, bool)
    a = np.exp(Ng)
    def acc(j, R, dR, frac):
        # interpolate coefficients between grid j and j+1
        e1 = eps1[j] + frac * (eps1[j + 1] - eps1[j])
        e2 = eps2[j] + frac * (eps2[j + 1] - eps2[j])
        laH = lnaH[j] + frac * (lnaH[j + 1] - lnaH[j])
        kk = np.exp(2 * (lnk_code - laH))
        return -(3 - e1 + e2) * dR - kk * R
    order = np.argsort(istart)
    jmin = istart.min()
    for j in range(jmin, len(Ng) - 1):
        new = (~on) & (istart <= j)
        if new.any():
            z = a[j] * np.sqrt(2 * eps1[j])
            kn = k[new]
            R[new] = 1 / (z * np.sqrt(2 * kn))
            dR[new] = -(1j * kn / np.exp(lnaH[j]) + 1 + eps2[j] / 2) * R[new]
            on |= new
        if not on.any():
            continue
        h = Ng[j + 1] - Ng[j]
        m = on
        r0, v0 = R[m], dR[m]
        lk = lnk_code[m]
        def f(frac, r, v):
            e1 = eps1[j] + frac * (eps1[j + 1] - eps1[j])
            e2 = eps2[j] + frac * (eps2[j + 1] - eps2[j])
            laH = lnaH[j] + frac * (lnaH[j + 1] - lnaH[j])
            return v, -(3 - e1 + e2) * v - np.exp(2 * (lk - laH)) * r
        k1r, k1v = f(0, r0, v0)
        k2r, k2v = f(0.5, r0 + 0.5 * h * k1r, v0 + 0.5 * h * k1v)
        k3r, k3v = f(0.5, r0 + 0.5 * h * k2r, v0 + 0.5 * h * k2v)
        k4r, k4v = f(1.0, r0 + h * k3r, v0 + h * k3v)
        R[m] = r0 + h / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
        dR[m] = v0 + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return k**3 * np.abs(R)**2 / (2 * np.pi**2)

if __name__ == "__main__":
    import time
    t = time.time()
    o = spectrum("staro", 0.0, 3.0, 0.1, nk=40)
    print(f"pure Starobinsky: N_*={o['Nst']:.2f}, n_s={o['ns']:.4f}, alpha_s={o['alpha_s']:.2e}, r={o['r']:.4f}, "
          f"V0={o['V0']:.3e} (mu={np.sqrt(o['V0']) * MPl_GeV:.3e} GeV), T_reh={o['Treh']:.2e} GeV, t={time.time() - t:.1f}s")
    print("  compare fixed background: N_*=51.0, n_s=0.9623, mu=3.20e13 GeV, T_reh=2.5e9 GeV")
    print("  P_R range:", o["P"].min(), o["P"].max())
