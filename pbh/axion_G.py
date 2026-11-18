r"""
Anharmonic energy function for an axion-like curvaton with V = m^2 f^2 (1 - cos theta).
Field equation in units t_hat = m t:
   MD (H = 2/(3t)):  theta'' + (2/t) theta' + sin theta = 0,  Omega = (f/M_P)^2 G_MD(theta_i)   (constant in MD)
   RD (H = 1/(2t)):  theta'' + (3/(2t)) theta' + sin theta = 0, Omega(t) = (f/M_P)^2 G_RD(theta_i) t_hat^{1/2}
with  G_MD = (3/4) lim rho_hat t_hat^2,  G_RD = (4/3) lim rho_hat t_hat^{3/2},  rho_hat = theta'^2/2 + 1 - cos theta.
Harmonic limit check: G_MD -> theta^2/6 (cf. Kawasaki-Kitajima-Yanagida arXiv:1207.2550, r ~ (1/6)(f theta/M_P)^2 T_R/T_dec).
Late-time limit taken as the average over the last few oscillations.
"""
from paths import PBH_ROOT, PBHBOUNDS, CEFFYL, NG15, NPROC
import numpy as np
from scipy.integrate import solve_ivp

def G_of(theta0, era, tmax_extra=400.0):
    p = 2.0 if era == "MD" else 1.5
    # series start at small t: theta ~ theta0 - sin(theta0) t^2 / (2(1+p))
    t0 = 1e-3
    th0 = theta0 - np.sin(theta0) * t0**2 / (2 * (1 + p))
    dth0 = -np.sin(theta0) * t0 / (1 + p)
    delay = max(0.0, -np.log(max(np.pi - theta0, 1e-300)))   # hilltop delay ~ log(1/(pi-theta))
    tmax = tmax_extra + 20 * delay
    sol = solve_ivp(lambda t, y: [y[1], -p / t * y[1] - np.sin(y[0])], (t0, tmax), [th0, dth0],
                    rtol=1e-10, atol=1e-12, dense_output=True, method="DOP853")
    tt = np.linspace(tmax - 60, tmax, 6000)
    th, dth = sol.sol(tt)
    rho = 0.5 * dth**2 + 1 - np.cos(th)
    if era == "MD":
        return 0.75 * np.mean(rho * tt**2)
    return (4 / 3) * np.mean(rho * tt**1.5)

def build_tables(n=160):
    th_lo = np.linspace(0.01, 2.5, 60)
    th_hi = np.pi - np.logspace(np.log10(np.pi - 2.5), -12, n)[1:]
    th = np.concatenate([th_lo, th_hi])
    GMD = np.array([G_of(x, "MD") for x in th])
    GRD = np.array([G_of(x, "RD") for x in th])
    np.savez(PBH_ROOT + "/axion_G.npz", th=th, GMD=GMD, GRD=GRD)
    return th, GMD, GRD

if __name__ == "__main__":
    th, GMD, GRD = build_tables()
    for x in [0.05, 0.5, 1.0, 2.0, 3.0, np.pi - 1e-3, np.pi - 1e-6, np.pi - 1e-10]:
        i = np.argmin(abs(th - x))
        print(f"theta={th[i]:.10f}  G_MD={GMD[i]:.4e}  G_MD/(theta^2/6)={GMD[i]/(th[i]**2/6):.3f}  G_RD={GRD[i]:.4e}  G_RD/G_RD,harm={GRD[i]/(GRD[0]*(th[i]/th[0])**2):.3f}")
