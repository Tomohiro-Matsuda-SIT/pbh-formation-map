"""Grid fits of the DW GW spectrum to PTA free spectra (pta_like.py).
  DW-only:      lnL_DW(T_ann, alpha)
  DW+SMBHB:     lnL_DWB = max_A lnL(SMBHB(A, gamma=13/3) + DW)   (SMBHB amplitude profiled)
  SMBHB-only:   lnL_B   = max_A lnL(SMBHB(A))
Outputs dw_pta_grid_{set}.npz."""
import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
import fopt_dw_pipeline as fd, pta_like as pl
name = sys.argv[1] if len(sys.argv) > 1 else "NG15"
F = pl.FreeSpec(name)
lT = np.linspace(np.log10(5e-3), 9.0, 451); la = np.linspace(-4, 0, 201)     # covers DW_BOX
TT, AA = np.meshgrid(lT, la, indexing="ij")
P = np.c_[TT.ravel(), AA.ravel()]
h2O = fd.dw_gw(*P.T)(F.f) * fd.h**2
LDW = F.lnL(h2O).reshape(TT.shape)
LDWB = pl.profile_smbhb(F, h2O).reshape(TT.shape)
LB = pl.profile_smbhb(F, np.zeros((1, len(F.f))))[0]
L0 = F.lnL0()
i = np.unravel_index(LDW.argmax(), LDW.shape); j = np.unravel_index(LDWB.argmax(), LDWB.shape)
print(f"[{name}, {len(F.f)} bins]  lnL relative to no signal:  SMBHB-only {LB - L0:.1f};  DW-only best {LDW.max() - L0:.1f} "
      f"at T={1e3 * 10**TT[i]:.0f} MeV, alpha={10**AA[i]:.3f};  DW+SMBHB best {LDWB.max() - L0:.1f} at T={1e3 * 10**TT[j]:.0f} MeV, alpha={10**AA[j]:.3f}")
m = LDW > LDW.max() - 3
print(f"   DW-only dlnL>-3 region: T [{1e3 * 10**TT[m].min():.0f}, {1e3 * 10**TT[m].max():.0f}] MeV, alpha [{10**AA[m].min():.3f}, {10**AA[m].max():.2f}]")
u = LDWB >= LB - 1.92
print(f"   'upper limit' region (DW+SMBHB not worse than SMBHB-only by >1.92): fraction of grid {u.mean():.3f}")
np.savez(f"dw_pta_grid_{name}.npz", lT=lT, la=la, LDW=LDW, LDWB=LDWB, LB=LB, L0=L0, Lmax=LDW.max())
