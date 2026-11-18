"""USR channel: sensitivity of the maximal prior probability per (M, f) cell to the prior on log10 A.
Alternative priors: (phi0, log10 s) uniform over the node grid; log10 A uniform within W decades below A_c."""
import numpy as np, json
import usr_pushforward as upf, results_store as rs
G0 = upf.gaussian_from_allowed("planck")
out = {}
nnode = len(upf.nodes)
for W in (3.0, 1.0, 0.3, 0.1, 0.01):
    G = dict(G0)
    # p(theta) such that p * dlogA * dph * dls integrates to (1/nnode)(dlogA/W)
    G["pdf"] = lambda th, W=W: np.full(len(th), 1.0 / (nnode * W * upf.dph * upf.dls))
    for ng in ("G", "NG"):
        res, _ = upf.pushforward("planck", ng, G)
        lM, lf, ok, det, w = res
        P, _ = rs.maps(dict(lM=lM, lf=lf, ok=ok, det=det, w=w))
        Ptot = float(np.sum(w * ok * (lf > -12)))
        out[f"W={W}_{ng}"] = dict(max_cell=float(P.max()), total_reach=Ptot, n_cells_ge_eps=int((P >= rs.EPS).sum()))
        print(W, ng, out[f"W={W}_{ng}"])
out["project_prior"] = {ng: json.load(open("results/usr_bump/planck_%s/meta.json" % ng))["stats"]["P_reach"] for ng in ("G", "NG")}
json.dump(out, open("usr_prior_sens.json", "w"), indent=1)
