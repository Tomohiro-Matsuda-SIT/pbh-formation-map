r"""
Standard-Model effective degrees of freedom g_rho(T), g_s(T): fitting functions of
Saikawa & Shirai, JCAP 05 (2018) 035, arXiv:1803.01038, Appendix C.
  T >= 120 MeV : g_rho = sum a_i t^i / sum b_i t^i ;  g_rho/g_s = 1 + sum c_i t^i / sum d_i t^i ,  t = ln(T/GeV)
  T <  120 MeV : sums of f_rho, b_rho, f_s, b_s, S_fit terms (their Eqs. C.3-C.4)
Coefficients transcribed from the arXiv PDF (text extraction); validated in check() against
g_rho,fin = 3.383, g_s,fin = 3.931 (their Eqs. 3.18-3.19) and continuity at 120 MeV.
"""
import numpy as np

a = [1, 1.11724E+00, 3.12672E-01, -4.68049E-02, -2.65004E-02, -1.19760E-03, 1.82812E-04, 1.36436E-04,
     8.55051E-05, 1.22840E-05, 3.82259E-07, -6.87035E-09]
b = [1.43382E-02, 1.37559E-02, 2.92108E-03, -5.38533E-04, -1.62496E-04, -2.87906E-05, -3.84278E-06,
     2.78776E-06, 7.40342E-07, 1.17210E-07, 3.72499E-09, -6.74107E-11]
c = [1, 6.07869E-01, -1.54485E-01, -2.24034E-01, -2.82147E-02, 2.90620E-02, 6.86778E-03, -1.00005E-03,
     -1.69104E-04, 1.06301E-05, 1.69528E-06, -9.33311E-08]
d = [7.07388E+01, 9.18011E+01, 3.31892E+01, -1.39779E+00, -1.52558E+00, -1.97857E-02, -1.60146E-01,
     8.22615E-05, 2.02651E-02, -1.82134E-05, 7.83943E-05, 7.13518E-05]
me, mmu, mpi0, mpic, m1, m2, m3, m4 = 511e-6, 0.1056, 0.135, 0.140, 0.5, 0.77, 1.2, 2.0

def _poly(co, t):
    return sum(ci * t**i for i, ci in enumerate(co))
def f_rho(x): return np.exp(-1.04855 * x) * (1 + 1.03757 * x + 0.508630 * x**2 + 0.0893988 * x**3)
def b_rho(x): return np.exp(-1.03149 * x) * (1 + 1.03317 * x + 0.398264 * x**2 + 0.0648056 * x**3)
def f_s(x):   return np.exp(-1.04190 * x) * (1 + 1.03400 * x + 0.456426 * x**2 + 0.0595248 * x**3)
def b_s(x):   return np.exp(-1.03365 * x) * (1 + 1.03397 * x + 0.342548 * x**2 + 0.0506182 * x**3)
def S_fit(x): return 1 + 7 / 4 * np.exp(-1.0419 * x) * (1 + 1.034 * x + 0.456426 * x**2 + 0.0595249 * x**3)

def _low_rho(T):
    return (2.030 + 1.353 * S_fit(me / T)**(4 / 3) + 3.495 * f_rho(me / T) + 3.446 * f_rho(mmu / T)
            + 1.05 * b_rho(mpi0 / T) + 2.08 * b_rho(mpic / T) + 4.165 * b_rho(m1 / T) + 30.55 * b_rho(m2 / T)
            + 89.4 * b_rho(m3 / T) + 8209 * b_rho(m4 / T))
def _low_s(T):
    return (2.008 + 1.923 * S_fit(me / T) + 3.442 * f_s(me / T) + 3.468 * f_s(mmu / T)
            + 1.034 * b_s(mpi0 / T) + 2.068 * b_s(mpic / T) + 4.16 * b_s(m1 / T) + 30.55 * b_s(m2 / T)
            + 90 * b_s(m3 / T) + 6209 * b_s(m4 / T))
def _high_rho(T):
    t = np.log(T); return _poly(a, t) / _poly(b, t)
def _high_s(T):
    t = np.log(T); return _high_rho(T) / (1 + _poly(c, t) / _poly(d, t))

TMAX = 1e16
def g_rho(T):
    T = np.asarray(T, float)
    Th = np.clip(T, 0.12, TMAX); Tl = np.clip(T, 1e-12, 0.12)
    return np.where(T >= 0.12, _high_rho(Th), _low_rho(Tl))
def g_s(T):
    T = np.asarray(T, float)
    Th = np.clip(T, 0.12, TMAX); Tl = np.clip(T, 1e-12, 0.12)
    return np.where(T >= 0.12, _high_s(Th), _low_s(Tl))

def check():
    rows = []
    for T in [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.1199, 0.12, 0.15, 0.2, 0.5, 1, 10, 100, 1e3, 1e6, 1e9, 1e15]:
        rows.append((T, float(g_rho(T)), float(g_s(T))))
    return rows

if __name__ == "__main__":
    for T, gr, gs in check():
        print(f"T={T:9.3g} GeV  g_rho={gr:8.3f}  g_s={gs:8.3f}")
    print("continuity at 120 MeV: rho", float(_low_rho(0.12)), float(_high_rho(0.12)),
          " s", float(_low_s(0.12)), float(_high_s(0.12)))
