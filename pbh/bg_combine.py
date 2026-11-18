r"""
Combine the background nodes (bg_node.py outputs) into hierarchical-prior results and store them as NEW conditions
'<cond>_bg<band initial>' (e.g. fopt/GV_dc0.5_bgP); fixed-background conditions are left untouched.

Hierarchical prior:  p(cell) = sum_{model in E,T} (1/2) * (1/K) sum_k p(cell | psi_k),   psi_k ~ pi(psi | model)
(psi = background, drawn from the truncated Gaussian of bg_prior.py).  Each sample of node k carries the weight
w = 1 / (2 K n_k).  A node on which the model has no allowed parameter region contributes nothing (its weight is
kept in the normalisation: the model cannot be realised on that background).
Also writes bg_diag.pdf: P_reach per node vs T_reh (one panel per model), and bg_diag.json.
usage: python bg_combine.py [band=planck]
"""
import os, sys, glob, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import results_store as rs
from paths import PBH_ROOT

band = sys.argv[1] if len(sys.argv) > 1 else "planck"
SUF = "_bg" + band[0].upper()
files = sorted(glob.glob(os.path.join(PBH_ROOT, "bgscan", band, "[ET]_[0-9]*.npz")))
nodes = []
for f in files:
    d = np.load(f)
    info = json.loads(str(d["info"]))
    nodes.append(dict(file=os.path.basename(f), ET=os.path.basename(f)[0], bg=info["bg"], info=info,
                      data={k.replace("__", "/"): d[k] for k in d.files if k != "info"}))
K = {m: sum(n["ET"] == m for n in nodes) for m in ("E", "T")}
nET = sum(1 for m in K if K[m] > 0)
print(f"{band}: {len(nodes)} nodes  {K}")

DESC = {"fopt": "supercooled FOPT", "dw": "biased domain walls", "fermiball": "Fermi-ball collapse",
        "infwall": "walls nucleated during inflation", "cstring": "cosmic-string loop collapse"}
conds = sorted({k for n in nodes for k in n["data"]})
diag = {}
for key in conds:
    model, cond = key.split("/")
    L = {q: [] for q in ("lM", "lf", "ok", "det", "w", "lTreh", "la", "wreh", "ET")}
    per = []
    for n in nodes:
        a = n["data"].get(key)
        b = n["bg"]
        if a is None:
            per.append((np.log10(b["Treh"]), n["ET"], 0.0, False)); continue
        m = a.shape[1]
        L["lM"].append(a[0]); L["lf"].append(a[1]); L["ok"].append(a[2] > 0); L["det"].append(a[3] > 0)
        L["w"].append(np.full(m, 1.0 / (nET * K[n["ET"]] * m)))
        for q, v in (("lTreh", np.log10(b["Treh"])), ("la", np.log10(b["alpha"])), ("wreh", b["w"]),
                     ("ET", 0.0 if n["ET"] == "E" else 1.0)):
            L[q].append(np.full(m, v, np.float32))
        per.append((np.log10(b["Treh"]), n["ET"], float(np.mean((a[1] > -12) & (a[2] > 0))), True))
    if not L["lM"]:
        continue
    A = {q: np.concatenate(v) for q, v in L.items()}
    st = rs.save(model, cond + SUF, A["lM"].astype(float), A["lf"].astype(float), A["ok"], A["det"],
                 f"{DESC[model]} '{cond}' on the alpha-attractor background family (E/T, {band} n_s band), "
                 f"hierarchical prior over (alpha, w_reh, T_reh); {len(nodes)} background nodes",
                 ["background.py: E/T alpha-attractor, slow roll, constant w_reh = (n-1)/(n+1), free T_reh",
                  "bg_prior.py: truncated-Gaussian prior on CMB-allowed psi (n_s 2 sigma, r < 0.036, T_reh > 5 MeV)",
                  "per-node model prior rebuilt with the project rule (bg_node.py)"],
                 ["background.py", "bg_prior.py", "bg_node.py", "bg_combine.py"],
                 extra=dict(w=A["w"], lTreh=A["lTreh"], la=A["la"], wreh=A["wreh"], ET=A["ET"]))
    diag[key] = dict(stats=st, nodes_present=int(sum(p[3] for p in per)), per_node=per)
    print(f"  {key + SUF:28s} P_reach={st['P_reach']:.3g}  P_reach_det={st['P_reach_det']:.3g}  "
          f"nodes with allowed region {diag[key]['nodes_present']}/{len(nodes)}")
json.dump(diag, open(os.path.join(PBH_ROOT, f"bg_diag_{band}.json"), "w"), indent=1, default=float)

# diagnostic: P_reach per node vs T_reh
show = [k for k in ("fopt/GV_dc0.5", "dw/ulL_Lt0.8_width", "dw/ptaL_Lt0.8_width", "fermiball/base",
                    "infwall/cen", "cstring/free") if k in diag]
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6), layout="constrained", sharex=True)
for ax, k in zip(axs.flat, show):
    for et, c, mk in (("E", "#0072B2", "o"), ("T", "#D55E00", "s")):
        P = [(x, p) for x, e, p, _ in diag[k]["per_node"] if e == et]
        if P:
            x, p = np.array(P).T
            ax.scatter(x, np.maximum(p, 1e-7), s=10, c=c, marker=mk, label=f"{et}-model")
    fixed = (json.load(open(os.path.join(PBH_ROOT, "results", "index.json"))).get(k) or {}).get("stats") or {}
    if fixed.get("P_reach") is not None:
        ax.axhline(max(fixed["P_reach"], 1e-7), color="0.3", ls="--", lw=0.8)
        ax.axvline(np.log10(2.5e9), color="0.6", ls=":", lw=0.8)
    ax.set_yscale("log"); ax.set_ylim(5e-8, 1.5)
    ax.set_title(k, fontsize=7.5, loc="left")
for ax in axs[1]:
    ax.set_xlabel(r"$\log_{10}(T_{\rm reh}/{\rm GeV})$")
for ax in axs[:, 0]:
    ax.set_ylabel(r"$P_{\rm reach}$ on the node")
axs[0, 0].legend(fontsize=6.5, frameon=False)
fig.suptitle(f"Per-background reach probability ({band} band); dashed: fixed Starobinsky, dotted: its $T_{{\\rm reh}}$",
             fontsize=8)
fig.savefig(os.path.join(PBH_ROOT, f"bg_diag_{band}.pdf")); fig.savefig(os.path.join(PBH_ROOT, f"bg_diag_{band}.png"), dpi=200)
print("wrote bg_diag_%s.pdf/json" % band)
