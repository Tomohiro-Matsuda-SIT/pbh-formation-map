r"""
Bayesian comparison of the interpretations of the NANOGrav 15-yr signal (referee response).
Data: NG15 HD free spectrum, 14 bins, KDE likelihood (pta_like.py).  Evidences Z = \int d theta pi(theta) L(theta)
by direct integration on grids; all priors uniform in the listed variables (stated explicitly because Bayes
factors scale with the prior volume):
   B     SMBHB power law, gamma = 13/3, log10 A in [-18, -11]
   Bg    SMBHB power law, gamma in [0, 7], log10 A in [-18, -11]
   DW    domain walls only: (log10 T_ann/GeV, log10 alpha_ann) in DW_BOX = [log10 5e-3, 9] x [-4, 0], restricted
         to the non-PTA conditions (alpha_ann < 1, sigma^{1/3} < T_reh, LVK O3)
   DW+B  domain walls + SMBHB (gamma = 13/3), product prior
PBH overproduction: the PBH limits are an additional likelihood factor 1_ok(theta; L/t) (ok flag of
fopt_dw_pipeline.dw_flags_cc, i.e. current PBH limits with the critical-collapse mass function), for
L/t = 0.5, 0.8, 1.2.  Output: dw_bayes.json.
Noise-model systematics are NOT covered: the free spectrum is the published one (Zenodo 10344086).
"""
import json, numpy as np, warnings
warnings.filterwarnings("ignore")
import fopt_dw_pipeline as fd, pta_like as pl
from paths import PBH_ROOT

F = pl.FreeSpec("NG15"); L0 = F.lnL0()
lA = np.linspace(-18, -11, 281)
PLs = pl.h2Omega_powerlaw(F.f, lA)                       # (nA, nf)
LB = F.lnL(PLs) - L0
lse = lambda x: float(np.log(np.mean(np.exp(x - x.max()))) + x.max())      # ln of the prior-averaged likelihood
res = {"lnZ_B": lse(LB), "B_bestfit_log10A": float(lA[LB.argmax()]), "B_lnLmax": float(LB.max())}
gam = np.linspace(0, 7, 141)
LBg = np.array([F.lnL(pl.h2Omega_powerlaw(F.f, lA, g)) - L0 for g in gam])
res["lnZ_Bg"] = lse(LBg.ravel()); i = np.unravel_index(LBg.argmax(), LBg.shape)
res["Bg_bestfit"] = [float(gam[i[0]]), float(lA[i[1]])]; res["Bg_lnLmax"] = float(LBg.max())

lT = np.linspace(fd.DW_BOX[0, 0], fd.DW_BOX[0, 1], 361); la = np.linspace(fd.DW_BOX[1, 0], fd.DW_BOX[1, 1], 161)
TT, AA = np.meshgrid(lT, la, indexing="ij"); P = np.c_[TT.ravel(), AA.ravel()]
Tann, a, tann, tdom, sigma = fd.dw_quantities(P)
Om = fd.dw_gw(*P.T)
base = (Tann > 5e-3) & (a < 1) & (sigma**(1 / 3) < fd.TREH) & np.all(Om(fd.f_LVK) < 5.8e-9, axis=-1)
h2O = Om(F.f) * fd.h**2
LDW = F.lnL(h2O) - L0
res["prior_fraction_of_box"] = float(base.mean())
res["DW_lnLmax"] = float(LDW[base].max()); j = np.argmax(np.where(base, LDW, -np.inf))
res["DW_bestfit"] = dict(T_ann_MeV=float(1e3 * Tann[j]), alpha_ann=float(a[j]))
res["lnZ_DW"] = lse(LDW[base])
# DW + SMBHB: marginalise the SMBHB amplitude on a coarser amplitude grid
lA2 = lA[::4]; PL2 = PLs[::4]
LDWB = np.full((len(lA2), base.sum()), -np.inf)
hb = h2O[base]
for k in range(len(lA2)):
    LDWB[k] = F.lnL(hb + PL2[k][None, :]) - L0
res["lnZ_DWB"] = lse(LDWB.ravel()); res["DWB_lnLmax"] = float(LDWB.max())
# PBH limits as an extra factor
for Lt in (0.5, 0.8, 1.2):
    out = [fd.dw_flags_cc(P[i:i + 20000], Lt) for i in range(0, len(P), 20000)]
    lMs, lf, ok, det, _ = (np.concatenate(z) for z in zip(*out))
    ok = ok.astype(bool)
    w = np.where(ok[base], LDW[base], -np.inf)
    lnZ = float(np.log(np.sum(np.exp(w - LDW[base].max())) / base.sum()) + LDW[base].max())
    post = np.exp(LDW[base] - LDW[base].max()); post /= post.sum()
    wb = np.where(ok[base][None, :], LDWB, -np.inf); m = LDWB.max()
    res[f"Lt{Lt}"] = dict(lnZ_DW_pbh=lnZ, post_frac_ok=float(post[ok[base]].sum()),
                          post_frac_f_gt_1=float(post[lf[base] > 0].sum()),
                          lnZ_DWB_pbh=float(np.log(np.sum(np.exp(wb - m)) / wb.size) + m))
res["lnBF"] = {"DW/B": res["lnZ_DW"] - res["lnZ_B"], "DW/Bg": res["lnZ_DW"] - res["lnZ_Bg"],
               "DWB/B": res["lnZ_DWB"] - res["lnZ_B"],
               **{f"DW_pbh_Lt{Lt}/B": res[f"Lt{Lt}"]["lnZ_DW_pbh"] - res["lnZ_B"] for Lt in (0.5, 0.8, 1.2)},
               **{f"DWB_pbh_Lt{Lt}/B": res[f"Lt{Lt}"]["lnZ_DWB_pbh"] - res["lnZ_B"] for Lt in (0.5, 0.8, 1.2)}}
# prior-range dependence: DW prior restricted to T_ann in [5 MeV, 10 GeV] (three instead of eleven decades)
nar = base & (P[:, 0] < 1.0)
res["lnBF"]["DW(T_ann<10GeV)/B"] = lse(LDW[nar]) - res["lnZ_B"]
json.dump(res, open(PBH_ROOT + "/dw_bayes.json", "w"), indent=1)
print(json.dumps(res, indent=1))
