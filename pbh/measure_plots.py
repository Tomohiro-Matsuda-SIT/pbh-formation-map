r"""Figures for measure_checks.py (reads measure_checks.npz / .json; no recomputation).
   fig_pmax.pdf       largest probability per square among the mechanisms (continuous), reference analysis
   fig_eps_curve.pdf  class-(A) fraction as a function of the threshold, for every variation of the distribution
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
PE = [pe.Stroke(linewidth=2.4, foreground="k"), pe.Normal()]
import results_store as rs
from paths import PBH_ROOT

SC = [("ul_opt", "(a) optimistic, PTA upper limit"), ("ul_con", "(b) conservative, PTA upper limit"),
      ("pta_opt", "(c) optimistic, walls produce PTA signal"), ("pta_con", "(d) conservative, walls produce PTA signal")]
VAR = [("main", "reference (Gaussian)", "k", "-", 1.6), ("ps0.5", r"spread $\times0.5$", "#0072B2", "--", 1.0),
       ("ps2", r"spread $\times2$", "#0072B2", "-", 1.0), ("flat", "uniform over allowed range", "#D55E00", "-", 1.6),
       ("narrow", "narrower parameter ranges", "#009E73", "--", 1.0), ("wide", "wider parameter ranges", "#009E73", "-", 1.0),
       ("bgP", r"$\alpha$-attractor models, free $T_{\rm reh}$", "#CC79A7", "-", 1.0)]

def main():
    z = np.load(os.path.join(PBH_ROOT, "measure_checks.npz"))
    r = json.load(open(os.path.join(PBH_ROOT, "measure_checks.json")))
    eps = np.array(r["eps_grid"]); excl = z["excl_d3"]
    fa = os.path.join(PBH_ROOT, "act_checks.npz"); ACT = np.load(fa) if os.path.exists(fa) else None
    # ---- continuous map
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.0), layout="constrained", sharex=True, sharey=True)
    norm = LogNorm(1e-9, 1e-1)
    for ax, (sc, ttl) in zip(axs.flat, SC):
        P = z[f"P_d3_main_{sc}"].astype(float)
        ax.pcolormesh(rs.xe, rs.ye, np.where(excl, 1.0, np.nan).T, cmap="Greys", vmin=0, vmax=6, rasterized=True)
        pc = ax.pcolormesh(rs.xe, rs.ye, np.ma.masked_where(excl | (P <= 0), P).T, cmap="viridis", norm=norm,
                           rasterized=True)
        for v, ls in ((1e-5, "-"),):
            cs = ax.contour(rs.xc, rs.yc, np.where(excl, 0, P).T, levels=[v], colors="w", linestyles=ls, linewidths=1.2)
            cs.set_path_effects(PE)
        if f"P_d3_flat_{sc}" in z.files:
            ax.contour(rs.xc, rs.yc, np.where(excl, 0, z[f"P_d3_flat_{sc}"]).T, levels=[1e-5], colors="#D55E00",
                       linestyles="--", linewidths=1.1)
        out = (~z[f"imgbg_{sc}"]) & ~excl
        ax.contourf(rs.xc, rs.yc, out.astype(float).T, levels=[0.5, 1.5], colors="none", hatches=["////"])
        ax.plot(rs.xc, np.clip(rs.ENV_CUR, -12, 0.5), color="0.25", lw=0.8)
        ax.set_xlim(-18, 6); ax.set_ylim(-12, 0); ax.set_title(ttl, fontsize=8.5, loc="left")
    for ax in axs[1]:
        ax.set_xlabel(r"$\log_{10}(M/M_\odot)$")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\log_{10} f_{\rm PBH}$")
    fig.colorbar(pc, ax=axs, label=r"largest probability per square among the mechanisms, $\max p_c$", shrink=0.8,
                 extend="both")
    fig.legend(handles=[Line2D([], [], color="w", lw=1.2, path_effects=PE, label=r"$\max p_c=10^{-5}$, reference distribution"),
                        Line2D([], [], color="#D55E00", lw=1.1, ls="--", label=r"$\max p_c=10^{-5}$, uniform distribution"),
                        Patch(fc="w", ec="k", lw=0.5, label="no sample in the reference analysis"),
                        Patch(fc="w", ec="k", lw=0.5, hatch="////", label="no sample in any variation"),
                        Patch(color="#d9d9d9", label="excluded by current limits, or evaporated")],
               loc="outside lower center", ncol=2, fontsize=7, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(PBH_ROOT, f"fig_pmax.{ext}"), dpi=200)
    # ---- A(eps)
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0), layout="constrained", sharex=True, sharey=True)
    for ax, (sc, ttl) in zip(axs.flat, SC):
        for v, lab, c, ls, lw in VAR:
            k = f"d3/{v}/{sc}"
            if k in r:
                ax.plot(eps, r[k]["A_curve"], color=c, ls=ls, lw=lw, label=lab)
        if ACT is not None:     # ACT-compatible inflation models (bg_act_combine.py), two islands
            for isl, lab, ls in (("actH", r"ACT $n_s$, $T_{\rm reh}\sim10^{14\text{--}15}$ GeV", (0, (5, 1))),
                                 ("actL", r"ACT $n_s$, $T_{\rm reh}\sim5$--$100$ MeV", (0, (1, 1)))):
                P = ACT[f"P_{isl}_{sc}"][~excl]
                ax.plot(eps, [(P < e).mean() for e in eps], color="#E69F00", ls=ls, lw=1.2, label=lab)
        ax.axvline(1e-5, color="0.6", lw=0.6, ls=":")
        ax.axhline(r[f"image/{sc}"]["outside_bg"], color="0.5", lw=0.6, ls="-.")
        ax.set_xscale("log"); ax.set_xlim(1e-7, 1e-2); ax.set_ylim(0, 1); ax.set_title(ttl, fontsize=8.5, loc="left")
    for ax in axs[1]:
        ax.set_xlabel(r"threshold $\epsilon$ (probability per square)")
    for ax in axs[:, 0]:
        ax.set_ylabel("fraction of allowed plane\nnot reached, (A)")
    h, l = axs[0, 0].get_legend_handles_labels()
    h.append(Line2D([], [], color="0.5", lw=0.6, ls="-.")); l.append("no sample in any variation")
    fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=7, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(PBH_ROOT, f"fig_eps_curve.{ext}"), dpi=200)

if __name__ == "__main__":
    main()
