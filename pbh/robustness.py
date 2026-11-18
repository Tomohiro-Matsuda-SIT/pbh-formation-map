r"""
Robustness of the (A)/(B1)/(B2) fractions of the combined map.
usage:  python robustness.py run [NSAMP=1000000]   -> bgscan/robust/<variant>.npz (fixed Starobinsky background)
        python robustness.py analyze             -> robustness.json, fig_robustness.pdf/png

Variants (each changes ONE ingredient relative to the main analysis):
  eps         EPS = 1e-6, 1e-4 instead of 1e-5 (recomputed from the stored results; no rerun)
  ps0.5, ps2  prior covariance x 0.25 / x 4 (width x 0.5 / x 2; centre = mean of the allowed region kept)
  narrow,wide box edges that are theory choices (bg_node.EDGE): beta/H max, log alpha_ann min, log eta_chi min,
              sigma^{1/3} max (walls), log G mu min
  bgP         alpha-attractor background family with free reheating (hierarchical prior, Planck band)
  base        the rerun pipeline itself at default settings (reference for the rerun variants: same code path,
              same sample size), compared with the stored main result as a consistency check.
Rerun variants recompute FOPT, DW, Fermi-ball, inflationary walls and cosmic strings; the curvaton family and the
USR channel are taken from the store (their per-cell probabilities are < 1e-6 except the axion-like curvaton,
whose total probability is 3.7e-5).
Monte Carlo uncertainty: Poisson bootstrap (each sample re-weighted by an independent Poisson(1) variate,
NBOOT replicates, all models resampled jointly); quoted as the 16-84% range of the fraction.
Robust classes: a cell is 'robustly (A)' if it is (A) in every variant (main, eps, prior width, box edges,
background family) and 'robustly reached' if it is reached in every variant.
"""
import os, sys, json, subprocess, numpy as np
from concurrent.futures import ThreadPoolExecutor
from paths import PBH_ROOT
sys_argv = sys.argv[:]; sys.argv = sys.argv[:1]
import fig_combined as fc          # SCEN (main), excluded mask
import results_store as rs
sys.argv = sys_argv

D = os.path.join(PBH_ROOT, "bgscan", "robust"); os.makedirs(D, exist_ok=True)
RERUN = {"base": {}, "ps0.5": {"PBH_PSCALE": "0.5"}, "ps2": {"PBH_PSCALE": "2"},
         "narrow": {"PBH_BOXVAR": "narrow"}, "wide": {"PBH_BOXVAR": "wide"},
         "flat": {"PBH_MEASURE": "flat"}}
BGM = ("fopt", "dw", "fermiball", "infwall", "cstring")
NBOOT = 20
rng = np.random.default_rng(11)

def run(ns):
    bgjson = os.path.join(PBH_ROOT, "bgscan", "validate", "E_staro.json")
    def one(v):
        env = dict(os.environ, PBH_BG=bgjson, PYTHONUTF8="1", OMP_NUM_THREADS="2", **RERUN[v])
        r = subprocess.run([sys.executable, "bg_node.py", os.path.join(D, f"{v}.npz"), str(ns)], env=env,
                           capture_output=True, text=True, cwd=PBH_ROOT)
        print(v, r.stdout.strip()[-200:] if r.returncode == 0 else "FAILED\n" + r.stderr[-1500:], flush=True)
    with ThreadPoolExecutor(len(RERUN)) as ex:
        list(ex.map(one, RERUN))

# ------------------------------------------------------------------ analysis
def classify(Rm, eps):
    """Rm: {model: list of (P, Pd) maps}. Same rules as fig_combined (union over conditions within a model)."""
    reached, Ptot, Pdet = [], 0, 0
    for maps in Rm.values():
        P = np.max([m[0] for m in maps], axis=0); Pd = np.max([m[1] for m in maps], axis=0)
        r = P >= eps; reached.append(r); Ptot = Ptot + P * r; Pdet = Pdet + Pd * r
    anyR = np.any(reached, axis=0)
    detect = Pdet >= 0.5 * np.maximum(Ptot, 1e-300)
    cls = np.where(fc.excluded, 0, np.where(~anyR, 1, np.where(detect, 2, 3)))
    allowed = ~fc.excluded
    classify.last = cls
    return {c: float((cls == v).sum() / allowed.sum()) for c, v in (("A", 1), ("B1", 2), ("B2", 3))}

def maps_of(d, wboot=None):
    """(P, Pd) maps of a result dict, optionally re-weighted by bootstrap weights."""
    if wboot is None:
        return rs.maps(d)
    w = d["w"] * wboot if "w" in d else wboot / len(d["lM"])
    return rs.maps(dict(d, w=w))

def load_source(variant):
    """{(model, cond): result dict} for every key used by the four scenarios."""
    keys = {k for s in fc.SCEN.values() for ks in s.values() for k in ks}
    src = {}
    if variant in RERUN:
        z = np.load(os.path.join(D, f"{variant}.npz"))
    for m, c in keys:
        if variant in RERUN and m in BGM:
            nm = f"{m}__{c}"
            a = z[nm] if nm in z.files else None
            src[(m, c)] = None if a is None else dict(lM=a[0].astype(float), lf=a[1].astype(float), ok=a[2] > 0, det=a[3] > 0)
        elif variant == "bgP" and m in BGM:
            try:
                src[(m, c)] = rs.load(m, c + "_bgP")
            except FileNotFoundError:
                src[(m, c)] = None
        else:
            src[(m, c)] = rs.load(m, c)
    return src

CLS = {}

def fractions(src, eps, boot=False, tag=None):
    out = {}
    cache = {}
    W = {k: (None if (v is None or not boot) else rng.poisson(1.0, len(v["lM"])).astype(float)) for k, v in src.items()}
    for sc, s in fc.SCEN.items():
        Rm = {}
        for m, ks in s.items():
            mm = []
            for k in ks:
                if src[k] is None:
                    continue
                if k not in cache:
                    cache[k] = maps_of(src[k], W[k])
                mm.append(cache[k])
            if mm:
                Rm[m] = mm
        out[f"{sc[0]}_{sc[1]}"] = classify(Rm, eps)
        if tag:
            CLS[(tag, f"{sc[0]}_{sc[1]}")] = classify.last.copy()
    return out

def analyze():
    res = {}
    plan = [("main", "store", 1e-5), ("eps1e-6", "store", 1e-6), ("eps1e-4", "store", 1e-4)]
    plan += [(v, v, 1e-5) for v in RERUN if os.path.exists(os.path.join(D, f"{v}.npz"))]
    plan += [("bgP", "bgP", 1e-5)]
    for name, var, eps in plan:
        src = load_source(var)
        cen = fractions(src, eps, tag=name)
        reps = [fractions(src, eps, boot=True) for _ in range(NBOOT)]
        res[name] = {sc: {c: dict(val=cen[sc][c], lo=float(np.percentile([r[sc][c] for r in reps], 16)),
                                  hi=float(np.percentile([r[sc][c] for r in reps], 84))) for c in cen[sc]}
                     for sc in cen}
        print(name, {sc: round(v["A"]["val"], 3) for sc, v in res[name].items()}, flush=True)
    # robust classes: (A) under EVERY variant except 'base' (which duplicates main) / reached under every variant
    names = [n for n, _, _ in plan if n != "base"]
    allowed = ~fc.excluded
    rob = {}
    for sc in res["main"]:
        nA = sum((CLS[(n, sc)] == 1).astype(int) for n in names)
        rob[sc] = dict(nA=nA, A_all=float(((nA == len(names)) & allowed).sum() / allowed.sum()),
                       A_any=float(((nA > 0) & allowed).sum() / allowed.sum()),
                       reach_all=float(((nA == 0) & allowed).sum() / allowed.sum()))
        print(sc, {k: round(v, 3) for k, v in rob[sc].items() if k != "nA"})
    res["robust"] = {sc: {k: v for k, v in r.items() if k != "nA"} | {"variants": names} for sc, r in rob.items()}
    json.dump(res, open(os.path.join(PBH_ROOT, "robustness.json"), "w"), indent=1)
    np.savez(os.path.join(PBH_ROOT, "robustness_cls.npz"), names=np.array(names),
             **{f"{n}__{sc}": CLS[(n, sc)] for (n, sc) in CLS})
    plot(res)
    plot_map(rob, len(names))

def plot_map(rob, nvar):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6), layout="constrained", sharex=True, sharey=True)
    cols = ["#d9d9d9", "#7FC97F", "#C7E9C0", "#FDE0B5", "#F5C77E", "#B8641B"]
    cmap = ListedColormap(cols)
    # 0 excluded, 1 reached in all, 2 (A) in < half, 3 (A) in >= half, 4 (A) in all but one, 5 (A) in all
    titles = {"ul_opt": "(a) optimistic, PTA upper limit", "ul_con": "(b) conservative, PTA upper limit",
              "pta_opt": "(c) optimistic, walls produce PTA signal",
              "pta_con": "(d) conservative, walls produce PTA signal"}
    for ax, sc in zip(axs.flat, ["ul_opt", "ul_con", "pta_opt", "pta_con"]):
        nA = rob[sc]["nA"]
        c = np.where(fc.excluded, 0, np.where(nA == 0, 1, np.where(nA == nvar, 5, np.where(nA == nvar - 1, 4,
                     np.where(nA >= nvar / 2, 3, 2)))))
        ax.pcolormesh(rs.xe, rs.ye, c.T, cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 6), 6), rasterized=True)
        ax.plot(rs.xc, np.clip(rs.ENV_CUR, -12, 0.5), color="0.25", lw=0.8)
        ax.set_xlim(-18, 6); ax.set_ylim(-12, 0)
        ax.set_title(titles[sc] + f"\nfraction never reached: {rob[sc]['A_all']:.2f}", fontsize=8, loc="left")
    for ax in axs[1]:
        ax.set_xlabel(r"$\log_{10}(M/M_\odot)$")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\log_{10} f_{\rm PBH}$")
    from matplotlib.patches import Patch
    lab = ["excluded by current limits, or evaporated", "reached in every variation",
           "(A) in fewer than half of the variations", "(A) in at least half of the variations",
           "(A) in all variations but one", "(A) in every variation (never reached)"]
    fig.legend(handles=[Patch(color=c_, label=l) for c_, l in zip(cols, lab)], loc="outside lower center", ncol=3,
               fontsize=7, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(PBH_ROOT, f"fig_robust_map.{ext}"), dpi=200)

def plot(res):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab = {"main": r"main (threshold $\epsilon=10^{-5}$)", "eps1e-6": r"threshold $\epsilon=10^{-6}$",
           "eps1e-4": r"threshold $\epsilon=10^{-4}$", "base": "main, recomputed",
           "ps0.5": r"spread of distribution $\times 0.5$", "ps2": r"spread of distribution $\times 2$",
           "narrow": "narrower parameter ranges", "wide": "wider parameter ranges",
           "bgP": r"$\alpha$-attractor models, free $T_{\rm reh}$"}
    names = [n for n in lab if n in res]
    scen = [("ul_opt", "(a) optimistic, PTA upper limit", "#0072B2", "o"),
            ("ul_con", "(b) conservative, PTA upper limit", "#D55E00", "s"),
            ("pta_opt", "(c) optimistic, walls produce PTA signal", "#009E73", "^"),
            ("pta_con", "(d) conservative, walls produce PTA signal", "#CC79A7", "D")]
    fig, ax = plt.subplots(figsize=(3.4, 3.6), layout="constrained")
    y = np.arange(len(names))[::-1]
    for j, (k, l, c, mk) in enumerate(scen):
        v = np.array([res[n][k]["A"]["val"] for n in names])
        lo = np.array([res[n][k]["A"]["lo"] for n in names]); hi = np.array([res[n][k]["A"]["hi"] for n in names])
        yy = y + (j - 1.5) * 0.17
        # Monte Carlo uncertainty = half of the bootstrap 16-84% spread, drawn symmetrically about the estimate
        # (re-weighting near the threshold shifts the replicates slightly; the spread, not the shift, is quoted)
        err = np.maximum(0.5 * (hi - lo), np.abs(0.5 * (hi + lo) - v))   # larger of half-spread and shift
        ax.errorbar(v, yy, xerr=err, fmt=mk, ms=3.5, color=c, lw=0.8, capsize=1.5, label=l)
    ax.set_yticks(y); ax.set_yticklabels([lab[n] for n in names], fontsize=7)
    ax.set_xlabel(r"fraction of allowed plane in class (A)", fontsize=7)
    ax.set_xlim(0, 1); ax.grid(axis="x", color="0.9", lw=0.5)
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=1)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(PBH_ROOT, f"fig_robustness.{ext}"), dpi=200)

if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000)
    elif sys.argv[1] == "plot":       # redraw both figures from robustness.json / robustness_cls.npz (no recomputation)
        res = json.load(open(os.path.join(PBH_ROOT, "robustness.json")))
        z = np.load(os.path.join(PBH_ROOT, "robustness_cls.npz"))
        names = [n for n in z["names"] if n != "base"]
        allowed = ~fc.excluded
        rob = {}
        for sc in ("ul_opt", "ul_con", "pta_opt", "pta_con"):
            nA = sum((z[f"{n}__{sc}"] == 1).astype(int) for n in names)
            rob[sc] = dict(nA=nA, A_all=float(((nA == len(names)) & allowed).sum() / allowed.sum()))
            assert abs(rob[sc]["A_all"] - res["robust"][sc]["A_all"]) < 1e-9
        plot(res)
        plot_map(rob, len(names))
    else:
        analyze()
