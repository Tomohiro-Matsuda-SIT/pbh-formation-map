"""Publication versions of figures for paper/ (labels without internal condition names); copies the others.
usage: python paper_figs.py   (after bg_combine.py, fig_combined.py, robustness.py analyze)"""
import os, json, shutil, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import PBH_ROOT

OUT = os.path.join(os.path.dirname(PBH_ROOT), "paper")
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8,
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True})

def fig_background(band="planck"):
    diag = json.load(open(os.path.join(PBH_ROOT, f"bg_diag_{band}.json")))
    idx = json.load(open(os.path.join(PBH_ROOT, "results", "index.json"), encoding="utf-8"))
    show = [("fopt/GV_dc0.5", "Phase transition"),
            ("dw/ulL_Lt0.8_width", "Domain walls, case (i)"),
            ("dw/ptaL_Lt0.8_width", "Domain walls, case (ii)"),
            ("fermiball/base", "Fermi balls"),
            ("infwall/cen", "Walls from inflation"),
            ("cstring/free", "String loops")]
    fig, axs = plt.subplots(3, 2, figsize=(3.4, 4.4), layout="constrained", sharex=True, sharey=True)
    for ax, (k, title) in zip(axs.flat, show):
        for et, c, mk in (("E", "#0072B2", "o"), ("T", "#D55E00", "s")):
            P = np.array([(x, p) for x, e, p, _ in diag[k]["per_node"] if e == et])
            if len(P):
                ax.scatter(P[:, 0], np.maximum(P[:, 1], 2e-6), s=3, c=c, marker=mk, lw=0, alpha=0.8,
                           label=f"{et}-model")
        ref = (idx.get(k) or {}).get("stats") or {}
        if ref.get("P_reach"):
            ax.axhline(ref["P_reach"], color="0.3", ls="--", lw=0.7)
        ax.axvline(np.log10(2.5e9), color="0.6", ls=":", lw=0.7)
        ax.set_yscale("log"); ax.set_ylim(1.5e-6, 1.5)
        ax.set_title(title, fontsize=6.5, loc="left", pad=2)
    for ax in axs[-1]:
        ax.set_xlabel(r"$\log_{10}(T_{\rm reh}/{\rm GeV})$")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$P_{\rm reach}$")
    axs[0, 0].legend(fontsize=5.5, frameon=False, loc="lower right", markerscale=2)
    fig.savefig(os.path.join(OUT, "fig_background.pdf"))
    fig.savefig(os.path.join(OUT, "fig_background.png"), dpi=200)

def fig_individual():
    """per-mechanism reach maps exactly as they enter Fig. 1 (max of p_c over the conditions of a scenario column);
    common colour scale.  Two figures: reference background, and the alpha-attractor family."""
    import sys; sys.argv = sys.argv[:1]
    import results_store as rs
    from matplotlib.colors import LogNorm
    def union(keys):
        P = None
        for k in keys:
            try:
                p = rs.maps(rs.load(*k))[0]
            except FileNotFoundError:
                continue
            P = p if P is None else np.maximum(P, p)
        return P
    lt = ("0.5", "0.8", "1.2")
    ref = [("Phase transition", [("fopt", f"GV_dc{d}") for d in ("0.4", "0.5", "0.667")]),
           ("Domain walls (i), all $L/t$", [("dw", f"ulL_Lt{x}_width") for x in lt]),
           ("Domain walls (i), $L/t=0.8$", [("dw", "ulL_Lt0.8_width")]),
           ("Domain walls (ii), all $L/t$", [("dw", f"ptaL_Lt{x}_width") for x in lt]),
           ("Domain walls (ii), $L/t=0.8$", [("dw", "ptaL_Lt0.8_width")]),
           ("Fermi balls", [("fermiball", "base")]),
           ("Axion-like curvaton", [("axion_curvaton", "opt")]),
           ("Walls from inflation, opt.", [("infwall", "opt")]),
           ("Walls from inflation, cons.", [("infwall", "con")]),
           ("String loops", [("cstring", "free")]),
           ("Inflaton bump, exp. tail", [("usr_bump", "planck_NG_fine")]),
           ("Inflaton bump, Gaussian", [("usr_bump", "planck_G_fine")])]
    bgp = [(t, [(m, c + "_bgP") for m, c in ks]) for t, ks in ref
           if ks[0][0] in ("fopt", "dw", "fermiball", "infwall", "cstring") and "all $L/t$" not in t]
    for name, panels, ncol in (("fig_individual_ref", ref, 4), ("fig_individual_bg", bgp, 4)):
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(7.0, 1.75 * nrow + 0.5), layout="constrained",
                                sharex=True, sharey=True)
        norm_ = LogNorm(vmin=1e-9, vmax=1e-1)
        for ax, (title, keys) in zip(axs.flat, panels):
            P = union(keys)
            ax.fill_between(rs.xc, np.clip(rs.ENV_CUR, -12, 0), 0, color="#d9d9d9", step="mid", zorder=0)
            if P is not None and (P > 0).any():
                pc = ax.pcolormesh(rs.xe, rs.ye, np.ma.masked_less_equal(P, 0).T, cmap="viridis", norm=norm_,
                                   rasterized=True)
                if (P >= rs.EPS).any():
                    ax.contour(rs.xc, rs.yc, (P >= rs.EPS).astype(float).T, levels=[0.5], colors="#D55E00",
                               linewidths=0.9)
            ax.plot(rs.xc, np.clip(rs.ENV_CUR, -12, 0.5), color="0.25", lw=0.6)
            ax.set_xlim(-18, 6); ax.set_ylim(-12, 0)
            ax.set_title("(" + "abcdefghijkl"[list(axs.flat).index(ax)] + ") " + title, fontsize=6.5, loc="left", pad=2)
        for ax in axs.flat[len(panels):]:
            ax.set_visible(False)
        for ax in axs[-1]:
            ax.set_xlabel(r"$\log_{10}(M/M_\odot)$", fontsize=7)
        for ax in axs[:, 0]:
            ax.set_ylabel(r"$\log_{10} f_{\rm PBH}$", fontsize=7)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm_, cmap="viridis"), ax=axs, shrink=0.6,
                     label=r"probability per square, $p_c$")
        # per-mechanism maps belong to the analysis notes (release/analysis_notes), not to the paper (2026-11-17)
        OUTN = os.path.join(os.path.dirname(PBH_ROOT), "release", "analysis_notes"); os.makedirs(OUTN, exist_ok=True)
        fig.savefig(os.path.join(OUTN, f"{name}.pdf")); fig.savefig(os.path.join(OUTN, f"{name}.png"), dpi=200)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_individual()
    fig_background()
    for f in ("fig_combined_AB.pdf", "fig_robust_map.pdf", "fig_pmax.pdf", "fig_eps_curve.pdf"):   # last two: measure_checks.py plot
        if os.path.exists(os.path.join(PBH_ROOT, f)):
            shutil.copy(os.path.join(PBH_ROOT, f), OUT)
    print("figures in", OUT)
