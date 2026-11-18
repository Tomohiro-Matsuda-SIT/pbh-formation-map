r"""
Sensitivity of the FOPT channel to the GW spectrum used for the LVK / NG15 conditions (referee response).
usage:  python fopt_gwvar.py run MODE [NSAMP]     MODE = env | LV3 | LV2   (sets PBH_FOPT_GW; see fopt_dw_pipeline.py)
        python fopt_gwvar.py analyze               -> fopt_gwvar.json
'run' rebuilds the allowed region, the Gaussian of the project rule and NSAMP samples for the non-covariant
treatment (delta_c = 0.4, 0.5, 0.667) and stores them in bgscan/robust/gw_<MODE>.npz (same format as bg_node.py).
'analyze' recomputes max p_c and the (A) fractions with the FOPT maps replaced (all other mechanisms from the
rerun reference 'base').
"""
import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")

def run(mode, ns):
    os.environ["PBH_FOPT_GW"] = mode
    import fopt_dw_pipeline as fd
    from paths import PBH_ROOT
    assert fd.FOPT_GW == mode
    fd.PTA_MODE = "L"; fd.rng = np.random.default_rng(7)
    mu, cov, acc, S, A = fd.gaussian_prior(fd.FOPT_BOX, fd.fopt_allowed, nsamp=ns)
    res = {}
    info = dict(mode=mode, acc=acc, mean=mu.tolist(), std=np.sqrt(np.diag(cov)).tolist(),
                lT_allowed=[float(A[:, 0].min()), float(A[:, 0].max())])
    # allowed range of log10 T_eq as a function of beta/H (shows where LVK and NG15 cut)
    edges = [3, 5, 8, 12, 20, 30]
    info["cuts"] = []
    P = fd.FOPT_BOX[:, 0] + (fd.FOPT_BOX[:, 1] - fd.FOPT_BOX[:, 0]) * fd.rng.random((400000, 2))
    ok = fd.fopt_allowed(P)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (P[:, 1] >= a) & (P[:, 1] < b)
        ex = np.sort(P[m & ~ok, 0])
        info["cuts"].append(dict(bH=[a, b], frac_allowed=float(ok[m].mean()),
                                 excluded_lT=[float(ex.min()), float(ex.max())] if len(ex) else None))
    for dc in (0.4, 0.5, 0.667):
        out = [fd.fopt_flags(S[i:i + 50000], dc) for i in range(0, len(S), 50000)]
        lM, lf, ok_, det = (np.concatenate(z) for z in zip(*out))
        res[f"fopt__GV_dc{dc}"] = np.vstack([lM, lf, ok_, det]).astype(np.float32)
        sel = (ok_ > 0) & (lf > -12)
        info[f"P_reach_dc{dc}"] = float(sel.mean())
        info[f"lM_5_95_dc{dc}"] = [float(x) for x in np.percentile(lM[sel], [5, 95])] if sel.any() else None
    np.savez_compressed(os.path.join(PBH_ROOT, "bgscan", "robust", f"gw_{mode}.npz"), info=json.dumps(info), **res)
    print(json.dumps(info, indent=1))

def analyze():
    import measure_checks as mc
    rb = mc.rb
    xe, ye, excl = mc.grid(1 / 3); allowed = ~excl
    XX = np.meshgrid(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), indexing="ij")[0]
    base = mc.load("base"); out = {}
    for mode in ("env", "LV3", "LV2"):
        f = os.path.join(rb.D, f"gw_{mode}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f); src = dict(base)
        for k in list(src):
            nm = f"{k[0]}__{k[1]}"
            if nm in z.files:
                a = z[nm]; src[k] = dict(lM=a[0], lf=a[1], ok=a[2] > 0, det=a[3] > 0)
        out[mode] = dict(info=json.loads(str(z["info"])))
        for sc in ("ul_opt", "pta_opt"):
            P, im, mech = mc.pmax(src, sc, xe, ye, {})
            i = mech.index("fopt")
            Pf = np.max([mc.pmap(src[k], xe, ye) for k in mc.SCEN[sc]["fopt"]], axis=0)
            r = (Pf >= 1e-5) & allowed
            out[mode][sc] = dict(A_1e6=float((P[allowed] < 1e-6).mean()), A_1e5=float((P[allowed] < 1e-5).mean()),
                                 A_1e4=float((P[allowed] < 1e-4).mean()),
                                 fopt_reach_M=[float(XX[r].min() - 1 / 6), float(XX[r].max() + 1 / 6)] if r.any() else None,
                                 fopt_any_M=[float(XX[(Pf > 0) & allowed].min() - 1 / 6), float(XX[(Pf > 0) & allowed].max() + 1 / 6)],
                                 fopt_p_median=float(np.median(Pf[r])) if r.any() else None)
        print(mode, {k: v for k, v in out[mode].items() if k != "info"}, flush=True)
    from paths import PBH_ROOT
    json.dump(out, open(os.path.join(PBH_ROOT, "fopt_gwvar.json"), "w"), indent=1)

if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1_000_000)
    else:
        analyze()
