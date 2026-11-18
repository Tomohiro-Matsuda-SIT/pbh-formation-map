"""Add DW critical-collapse-width flags (DW_{case}_cc_{Lt}) to fopt_dw_samples.npz from stored prior samples."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
import fopt_dw_pipeline as fd
d = dict(np.load("fopt_dw_samples.npz"))
for case in ("ul", "pta"):
    S = d[f"DW_{case}_S"]
    for Lt in (0.5, 0.8, 1.2):
        out = [fd.dw_flags_cc(S[i:i + 50000], Lt) for i in range(0, len(S), 50000)]
        lMs, lf, ok, det, sD = (np.concatenate(z) for z in zip(*out))
        mono = d[f"DW_{case}_x_{Lt}"]
        pb = lf > -12
        print(f"{case} L/t={Lt}: s_D median {np.median(sD[pb]) if pb.any() else np.nan:.4f}; "
              f"P(f>1e-12 & ok) mono {np.mean(pb & (mono[2] > 0)):.4f} -> width {np.mean(pb & ok):.4f};"
              f"  P(ok&det) mono {np.mean(pb & (mono[2] > 0) & (mono[3] > 0)):.4f} -> {np.mean(pb & ok & det):.4f};"
              f"  shift logM* {np.median(lMs[pb] - mono[0][pb]) if pb.any() else np.nan:.2f}")
        d[f"DW_{case}_cc_{Lt}"] = np.vstack([lMs, lf, ok, det])
np.savez("fopt_dw_samples.npz", **d)
