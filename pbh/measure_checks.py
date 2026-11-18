r"""
Dependence of the combined map on the probability measure, the threshold and the square size (referee response).
usage:  python measure_checks.py          -> measure_checks.json, measure_checks.npz (maps), printed summary
        python measure_checks.py plot     -> fig_pmax.pdf/png, fig_eps_curve.pdf/png   (from the stored npz/json)

For every scenario of fig_combined.SCEN and every variant we compute, per square c,
    Pmax(c) = max over mechanisms (and over the treatments of a mechanism) of p_c            [Eq. (1) of the paper]
so that the class-(A) fraction at ANY threshold is  A(eps) = fraction of allowed squares with Pmax < eps.
Variants (same samples as robustness.py):
    main    stored main analysis (Gaussian with the moments of the allowed region R, truncated to R)
    base    the same, recomputed with the rerun code         ps0.5, ps2   Gaussian spread x0.5, x2
    flat    UNIFORM over R in the box coordinates (logarithmic parameters -> log-uniform)  [bg_node.py PBH_MEASURE=flat]
    narrow, wide   theory-chosen box edges moved             flatwide     uniform over R with the wide edges
    bgP     alpha-attractor family with free reheating temperature
Image of R ("can be produced at all"): a square is in the image if ANY sample of ANY variant on the reference
inflation model lands in it with ok = True (independent of the measure up to the sampling resolution: the
uniform variants have 1e6 samples per mechanism; FOPT and DW have two parameters, so R is covered at ~1e-3 of
its extent per axis).  image+bg additionally includes the alpha-attractor family.
Square size: d = 1/6, 1/3 (reference), 2/3 dex, with the threshold either fixed per square (eps) or fixed per
unit area (eps * (3 d)^2).
"""
import os, sys, json, numpy as np
import robustness as rb
fc, rs = rb.fc, rb.rs
from paths import PBH_ROOT

SC = ["ul_opt", "ul_con", "pta_opt", "pta_con"]
SCEN = {f"{k[0]}_{k[1]}": v for k, v in fc.SCEN.items()}
VARIANTS = ["main", "base", "ps0.5", "ps2", "flat", "narrow", "wide", "flatwide", "bgP"]
REF_BG = ["main", "base", "ps0.5", "ps2", "flat", "narrow", "wide", "flatwide"]   # reference inflation model
EPSG = np.logspace(-7, -1, 61)

def grid(d):
    xe = np.arange(-18, 6 + 1e-9, d); ye = np.arange(-12, 0 + 1e-9, d)
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    env = np.full(len(xc), np.inf)
    for nm in rs.CURRENT:
        lm, lf = rs._load(nm); o = np.argsort(lm); lm, lf = lm[o], lf[o]
        ins = (xc >= lm.min()) & (xc <= lm.max())
        env[ins] = np.minimum(env[ins], np.interp(xc[ins], lm, lf))
    XX, YY = np.meshgrid(xc, yc, indexing="ij")
    return xe, ye, (YY > env[:, None]) | (XX < rs.MEVAP)

def pmap(res, xe, ye):
    lM, lf = np.asarray(res["lM"], float), np.asarray(res["lf"], float)
    m = np.isfinite(lM) & np.isfinite(lf) & res["ok"].astype(bool)
    w = np.asarray(res["w"], float)[m] if "w" in res else np.full(m.sum(), 1.0 / max(len(lM), 1))
    return np.histogram2d(lM[m], lf[m], bins=[xe, ye], weights=w)[0]

def load(variant):
    keys = {k for s in SCEN.values() for ks in s.values() for k in ks}
    f = os.path.join(rb.D, f"{variant}.npz")
    z = np.load(f) if os.path.exists(f) else None
    if variant not in ("main", "bgP") and z is None:
        return None
    src = {}
    for m, c in keys:
        if z is not None and m in rb.BGM:
            nm = f"{m}__{c}"
            a = z[nm] if nm in z.files else None
            src[(m, c)] = None if a is None else dict(lM=a[0], lf=a[1], ok=a[2] > 0, det=a[3] > 0)
        elif variant == "bgP" and m in rb.BGM:
            try:
                src[(m, c)] = rs.load(m, c + "_bgP")
            except FileNotFoundError:
                src[(m, c)] = None
        else:
            src[(m, c)] = rs.load(m, c)
    return src

def pmax(src, sc, xe, ye, cache):
    """-> Pmax map and index of the mechanism that gives it."""
    mech = list(SCEN[sc]); out = []
    for m in mech:
        P = np.zeros((len(xe) - 1, len(ye) - 1))
        for k in SCEN[sc][m]:
            if src[k] is None:
                continue
            if k not in cache:
                cache[k] = pmap(src[k], xe, ye)
            P = np.maximum(P, cache[k])
        out.append(P)
    out = np.array(out)
    return out.max(0), out.argmax(0), mech

def run():
    res, store = {}, {}
    for d, tag in ((1 / 3, "d3"), (1 / 6, "d6"), (2 / 3, "d1.5")):
        xe, ye, excl = grid(d); allowed = ~excl
        store[f"excl_{tag}"] = excl
        for v in VARIANTS:
            if tag != "d3" and v not in ("main", "flat"):
                continue
            src = load(v)
            if src is None:
                print("missing variant", v); continue
            cache = {}
            for sc in SC:
                P, im, mech = pmax(src, sc, xe, ye, cache)
                store[f"P_{tag}_{v}_{sc}"] = P.astype(np.float32); store[f"im_{tag}_{v}_{sc}"] = im.astype(np.int8)
                A = [(P[allowed] < e).mean() for e in EPSG]
                res[f"{tag}/{v}/{sc}"] = dict(A_curve=[float(a) for a in A], mech=mech,
                                             A_1e5=float((P[allowed] < 1e-5).mean()),
                                             A_area=float((P[allowed] < 1e-5 * (3 * d)**2).mean()),
                                             zero=float((P[allowed] <= 0).mean()))
    # image of R (reference inflation model) and with the background family
    xe, ye, excl = grid(1 / 3); allowed = ~excl; xc = 0.5 * (xe[1:] + xe[:-1])
    for sc in SC:
        have = [v for v in REF_BG if f"P_d3_{v}_{sc}" in store]
        img = np.any([store[f"P_d3_{v}_{sc}"] > 0 for v in have], axis=0)
        imgbg = img | (store[f"P_d3_bgP_{sc}"] > 0)
        store[f"img_{sc}"] = img; store[f"imgbg_{sc}"] = imgbg
        col = (img & allowed).any(1); colbg = (imgbg & allowed).any(1)
        res[f"image/{sc}"] = dict(variants=have, outside=float((~img)[allowed].mean()),
                                  outside_bg=float((~imgbg)[allowed].mean()),
                                  Mmax=float(xe[1:][col].max()) if col.any() else None,
                                  Mmax_bg=float(xe[1:][colbg].max()) if colbg.any() else None)
    res["eps_grid"] = EPSG.tolist()
    json.dump(res, open(os.path.join(PBH_ROOT, "measure_checks.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(PBH_ROOT, "measure_checks.npz"), **store)
    for sc in SC:
        print(f"== {sc}")
        for v in VARIANTS:
            r = res.get(f"d3/{v}/{sc}")
            if r:
                c = r["A_curve"]
                print(f"  {v:9s} A(1e-7,1e-6,1e-5,1e-4,1e-3) = " + " ".join(f"{c[i]:.3f}" for i in (0, 10, 20, 30, 40))
                      + f"   no sample: {r['zero']:.3f}")
        for tag in ("d6", "d1.5"):
            for v in ("main", "flat"):
                r = res.get(f"{tag}/{v}/{sc}")
                if r:
                    print(f"  {tag:5s}{v:5s} A(eps=1e-5 per square) = {r['A_1e5']:.3f}   A(same density) = {r['A_area']:.3f}")
        print("  image:", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in res[f"image/{sc}"].items() if k != "variants"})

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        import measure_plots; measure_plots.main()
    else:
        run()
