r"""
Checks for the Kohri-consistency revision (2026-11-16) on the combined map of fig_combined.py.
 (1) class of the allowed cells at 1e-4 - 1e-2 Msun (mass range of the SIGW interpretation of the PTA signal,
     Kohri & Terada arXiv:2009.11853, Inomata, Kohri & Terada arXiv:2306.17834) in each scenario;
 (2) (A)/(B1)/(B2) fractions when the axion-like curvaton enters with the collapse-threshold band
     zeta_c = 1.0, 0.67, 0.5 (max over the band, N = 1e6; axion_zc.py) instead of zeta_c = 1.0 (3e5 samples);
 (3) the same with the infwall conditions on the background family replaced by the 'collapse before reheating'
     treatment (infwall_reh.py) -- bgP variant only.
usage: python kohri_checks.py [main|bgP]
"""
import sys, json, copy, numpy as np
VAR = sys.argv[1] if len(sys.argv) > 1 else "main"
sys.argv = [sys.argv[0]] + ([VAR] if VAR != "main" else [])
import fig_combined as fc
import results_store as rs


def classify(scen):
    R = {m: fc.reach(keys) for m, keys in scen.items()}
    anyR = np.any([R[k][0] for k in R], axis=0)
    Ptot = sum(R[k][1] * R[k][0] for k in R); Pdet = sum(R[k][2] * R[k][0] for k in R)
    detect = Pdet >= 0.5 * np.maximum(Ptot, 1e-300)
    cls = np.where(fc.excluded, 0, np.where(~anyR, 1, np.where(detect, 2, 3)))
    allowed = ~fc.excluded
    fr = {c: round(float((cls == v).sum() / allowed.sum()), 4) for c, v in (("A", 1), ("B1", 2), ("B2", 3))}
    return cls, fr, R


out = {}
for key, scen in fc.SCEN.items():
    tag = f"{key[0]}_{key[1]}"
    cls, fr, R = classify(scen)
    band = (fc.XX >= -4) & (fc.XX <= -2) & ~fc.excluded
    out[tag] = dict(base=fr, sigw_band={c: int(((cls == v) & band).sum()) for c, v in (("A", 1), ("B1", 2), ("B2", 3))},
                    sigw_band_reached_by={m: int((R[m][0] & band).sum()) for m in R})
    if key[1] == "opt":
        s2 = copy.deepcopy(scen)
        s2["axion"] = [("axion_curvaton", f"opt_zc{z}") for z in ("1.0", "0.67", "0.5")]
        cls2, fr2, R2 = classify(s2)
        out[tag]["axion_zc_band"] = fr2
        out[tag]["axion_cells_reached"] = int((R2["axion"][0] & ~fc.excluded).sum())
        out[tag]["axion_cells_new"] = int(((cls2 != 1) & (cls == 1)).sum())
    if VAR == "bgP":
        s3 = copy.deepcopy(scen)
        s3["infwall"] = [("infwall", c.replace("_bgP", "_rehc_bgP")) for _, c in scen["infwall"]]
        cls3, fr3, R3 = classify(s3)
        out[tag]["infwall_collapse"] = fr3
        out[tag]["infwall_cells_reached"] = [int((R[m][0] & ~fc.excluded).sum()) for m in ("infwall",)] + \
                                            [int((R3["infwall"][0] & ~fc.excluded).sum())]
    print(tag, out[tag], flush=True)
json.dump(out, open(f"kohri_checks_{VAR}.json", "w"), indent=1)
