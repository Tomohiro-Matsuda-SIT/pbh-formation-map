r"""
Combined result on the ACT-compatible inflation models (islands actH / actL of bg_prior_act.py; referee response).
usage: python bg_act_combine.py        -> act_checks.json, act_checks.npz
For each island all nodes get equal weight (a node on which a mechanism has no allowed parameters contributes
zero but keeps its weight, as in bg_combine.py).  Mechanisms rerun on these inflation models: FOPT, domain walls,
Fermi balls, walls from inflation, string loops.  The curvaton models (computed on the reference model only)
and the inflaton bump (a bump on the Starobinsky potential, excluded by the ACT n_s) are left out.
"""
import os, glob, json, numpy as np
import measure_checks as mc
rb = mc.rb
from paths import PBH_ROOT

xe, ye, excl = mc.grid(1 / 3); allowed = ~excl
XX = np.meshgrid(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), indexing="ij")[0]
out, store = {}, {}
for isl in ("actH", "actL"):
    files = sorted(glob.glob(os.path.join(PBH_ROOT, "bgscan", isl, "*.npz")))
    if not files:
        continue
    N = len(files); maps = {}; preach = {}; treh = []; ns = []
    for f in files:
        z = np.load(f); info = json.loads(str(z["info"]))
        treh.append(info["bg"]["Treh"]); ns.append(info["bg"]["ns"])
        for nm in z.files:
            if nm == "info":
                continue
            a = z[nm]; n = a.shape[1]
            d = dict(lM=a[0].astype(float), lf=a[1].astype(float), ok=a[2] > 0, det=a[3] > 0, w=np.full(n, 1.0 / (N * n)))
            maps[nm] = maps.get(nm, 0) + mc.pmap(d, xe, ye)
            preach[nm] = preach.get(nm, 0) + float(((a[2] > 0) & (a[1] > -12) & np.isfinite(a[0])).sum() / (N * n))
    r = dict(n_nodes=N, log10Treh=[float(np.log10(min(treh))), float(np.log10(max(treh)))],
             ns=[float(min(ns)), float(max(ns))], P_reach={k: v for k, v in sorted(preach.items())})
    for sc in mc.SC:
        mech = [m for m in mc.SCEN[sc] if m in rb.BGM]
        Pm = []
        for m in mech:
            P = np.zeros_like(XX)
            for (a, c) in mc.SCEN[sc][m]:
                if f"{a}__{c}" in maps:
                    P = np.maximum(P, maps[f"{a}__{c}"])
            Pm.append(P)
        Pm = np.array(Pm); P = Pm.max(0)
        store[f"P_{isl}_{sc}"] = P.astype(np.float32)
        r[sc] = dict(A_1e6=float((P[allowed] < 1e-6).mean()), A_1e5=float((P[allowed] < 1e-5).mean()),
                     A_1e4=float((P[allowed] < 1e-4).mean()), zero=float((P[allowed] <= 0).mean()),
                     per_mech={m: dict(cells_1e5=int(((Pm[i] >= 1e-5) & allowed).sum()), pmax=float((Pm[i] * allowed).max()),
                                       M=[float(XX[(Pm[i] >= 1e-6) & allowed].min()), float(XX[(Pm[i] >= 1e-6) & allowed].max())]
                                       if ((Pm[i] >= 1e-6) & allowed).any() else None) for i, m in enumerate(mech)})
    out[isl] = r
    print(isl, json.dumps(r, indent=1))
json.dump(out, open(os.path.join(PBH_ROOT, "act_checks.json"), "w"), indent=1)
np.savez_compressed(os.path.join(PBH_ROOT, "act_checks.npz"), **store)
