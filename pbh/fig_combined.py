"""
Combined (A)/(B) map on the (M, f_PBH) plane, built ONLY from the per-(model, condition) result store
(results/<model>/<condition>/samples.npz; see results_store.py, build_results.py).

Classes per 1/3-dex cell (grid and EPS = 1e-5 from results_store):
   excluded : above the envelope of current bounds (PBHbounds, commit 96bc091) or M < 5e14 g (evaporated)
   (A)      : allowed, no included result reaches the cell with prior probability >= EPS
   (B1)     : reached, and >= 50% of the reaching probability is detectable by projected probes
   (B2)     : reached, below projected reach
'Reach' counts a sample only if it passes current bounds with its own mass function (ok flag).
Within one model the reach is the union (max probability) over the conditions listed for the scenario
(e.g. the delta_c band of FOPT); across models the reaches are combined.

Inflaton (USR bump) channel is included in every scenario (Planck CMB band; optimistic = USR exponential
tail, conservative = Gaussian); its maximal prior probability per cell is ~1e-8 << EPS, so it reaches no cell
(see results/usr_bump/*/map.pdf).  Under the ACT band its allowed region is empty.
(2026-11-17: the USR results use the fine grid of 88 bump positions, usr_bump/*_fine.)
SCENARIOS (edit here; every entry refers to a stored result):
   columns: theory treatment (optimistic / conservative);  rows: PTA interpretation for the DW model
   (upper limit with SMBHB / DW explains the signal); all GW limits use the NANOGrav 15-yr likelihood.
usage: python fig_combined.py [variant]   variants: (none) main | mono (DW monochromatic) | epta (EPTA for
       the 'explains' row) | legacy (pointwise PTA / 1/yr band, monochromatic DW)
"""
import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.ndimage import binary_dilation
import results_store as rs

VAR = sys.argv[1] if len(sys.argv) > 1 else "main"
MF = "mono" if VAR in ("mono", "legacy") else "width"
DW_UL = "ul" if VAR == "legacy" else "ulL"
DW_EX = {"legacy": "pta", "epta": "ptaLE"}.get(VAR, "ptaL")
LT_ALL = ("0.5", "0.8", "1.2")

def dw(case, lts):
    return [("dw", f"{case}_Lt{lt}_{MF}") for lt in lts]

COMMON = {"curvaton_std": [("curvaton_std", f"zetac{z}") for z in ("0.7", "1.0", "1.3")],
          "inflating_curvaton": [("inflating_curvaton", "opt"), ("inflating_curvaton", "con")],
          "fermiball": [("fermiball", "base")]}
SCEN = {
    ("ul", "opt"):  {**COMMON, "fopt": [("fopt", f"GV_dc{d}") for d in ("0.4", "0.5", "0.667")],
                     "dw": dw(DW_UL, LT_ALL), "axion": [("axion_curvaton", "opt")],
                     "infwall": [("infwall", "opt")], "cstring": [("cstring", "free")],
                     "usr": [("usr_bump", "planck_NG_fine")]},
    ("ul", "con"):  {**COMMON, "fopt": [("fopt", f"cov_dc{d}") for d in ("0.4", "0.5", "0.667")],
                     "dw": dw(DW_UL, ("0.8",)), "axion": [("axion_curvaton", "con")],
                     "infwall": [("infwall", "con")], "cstring": [("cstring", "CC")],
                     "usr": [("usr_bump", "planck_G_fine")]},
}
SCEN[("pta", "opt")] = {**SCEN[("ul", "opt")], "dw": dw(DW_EX, LT_ALL)}
SCEN[("pta", "con")] = {**SCEN[("ul", "con")], "dw": dw(DW_EX, ("0.8",))}
if VAR == "bgP":
    # alpha-attractor background family + free reheating (hierarchical prior, Planck band; bg_combine.py).
    # Models whose result depends on the background through T_reh / H(N) / T_max use the '_bgP' conditions;
    # the curvaton family and the USR channel stay on the fixed Starobinsky background (not rerun).
    BGM = ("fopt", "dw", "fermiball", "infwall", "cstring")
    for s in SCEN.values():
        for m in BGM:
            s[m] = [(a, c + "_bgP") for a, c in s[m]]
CONTOURS = [("fopt", "--", "region reached by the phase transition"), ("dw", "-", "region reached by domain walls"),
            ("fermiball", "-.", "region reached by Fermi balls")]

_cache = {}
def getmaps(key):
    if key not in _cache:
        _cache[key] = rs.maps(rs.load(*key))
    return _cache[key]

def reach(keys):
    M = [getmaps(k) for k in keys]
    P = np.max([m[0] for m in M], axis=0); Pd = np.max([m[1] for m in M], axis=0)
    return P >= rs.EPS, P, Pd

XX, YY = np.meshgrid(rs.xc, rs.yc, indexing="ij")
excluded = (YY > rs.ENV_CUR[:, None]) | (XX < rs.MEVAP)
cmap = ListedColormap(["#d9d9d9", "#F5C77E", "#56B4E9", "#7FC97F"])   # excluded, A, B1, B2

if __name__ == "__main__":
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.8), layout="constrained", sharex=True, sharey=True)
    stats = {}
    for i, pta in enumerate(["ul", "pta"]):
        for j, th in enumerate(["opt", "con"]):
            ax = axs[i, j]
            R = {m: reach(keys) for m, keys in SCEN[(pta, th)].items()}
            anyR = np.any([R[k][0] for k in R], axis=0)
            Ptot = sum(R[k][1] * R[k][0] for k in R); Pdet = sum(R[k][2] * R[k][0] for k in R)
            detect = Pdet >= 0.5 * np.maximum(Ptot, 1e-300)
            cls = np.where(excluded, 0, np.where(~anyR, 1, np.where(detect, 2, 3)))
            ax.pcolormesh(rs.xe, rs.ye, cls.T, cmap=cmap, vmin=-0.5, vmax=3.5, edgecolors="face", linewidth=0,
                          rasterized=True)
            if th == "opt":   # axion-like curvaton: support of all samples (prob. per cell < EPS)
                a = rs.load("axion_curvaton", "opt")
                sup = binary_dilation(rs.pmap(a["lM"], a["lf"]) > 0, iterations=1)
                ax.contour(rs.xc, rs.yc, sup.astype(float).T, levels=[0.5], colors="#882255", linestyles=":",
                           linewidths=1.3)
            for key, ls, _ in CONTOURS:
                if R[key][0].any():
                    ax.contour(rs.xc, rs.yc, (R[key][0] & ~excluded).astype(float).T, levels=[0.5], colors="k",
                               linestyles=ls, linewidths=1.1)
            ax.plot(rs.xc, np.clip(rs.ENV_CUR, -12, 0.5), color="0.25", lw=0.8)
            ax.plot(rs.xc, np.clip(rs.ENV_FUT, -12, 0.5), color="0.25", lw=0.8, ls=(0, (2, 2)))
            allowed = ~excluded
            stats[(pta, th)] = {c: round(float((cls == v).sum() / allowed.sum()), 3) for c, v in (("A", 1), ("B1", 2), ("B2", 3))}
            ax.set_xlim(-18, 6); ax.set_ylim(-12, 0)
            ttl = {"opt": "optimistic", "con": "conservative"}[th]
            ptl = {"ul": "PTA upper limit", "pta": "walls produce PTA signal"}[pta]
            ax.set_title(f"({'abcd'[2 * i + j]}) {ttl}, {ptl}", fontsize=8.5, loc="left")
    for ax in axs[1]:
        ax.set_xlabel(r"$\log_{10}(M/M_\odot)$")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\log_{10} f_{\rm PBH}$")
    handles = [Patch(color="#d9d9d9", label="excluded by current limits, or evaporated"),
               Patch(color="#F5C77E", label="(A) allowed, produced by no mechanism"),
               Patch(color="#56B4E9", label="(B1) produced, testable by planned observations"),
               Patch(color="#7FC97F", label="(B2) produced, beyond planned sensitivity")]
    handles += [Line2D([], [], color="k", ls=ls, label=lab) for _, ls, lab in CONTOURS]
    handles += [Line2D([], [], color="#882255", ls=":", lw=1.3,
                       label=r"range of axion-like curvaton (below $10^{-5}$ per square)"),
                Line2D([], [], color="0.25", lw=0.8, label="strongest current limit"),
                Line2D([], [], color="0.25", lw=0.8, ls=(0, (2, 2)), label="strongest planned limit")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=7, frameon=False)
    out = "fig_combined_AB" + ("" if VAR == "main" else f"_{VAR}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    import json
    json.dump({f"{k[0]}_{k[1]}": v for k, v in stats.items()} | {"variant": VAR, "scenarios":
              {f"{k[0]}_{k[1]}": {m: ["/".join(x) for x in v] for m, v in s.items()} for k, s in SCEN.items()}},
              open(f"{out}_stats.json", "w"), indent=1)
    for k, v in stats.items():
        print(k, v)
