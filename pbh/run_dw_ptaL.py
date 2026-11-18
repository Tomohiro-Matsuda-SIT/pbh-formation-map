"""DW with PTA free-spectrum likelihoods: new prior, monochromatic and critical-collapse-width flags.
usage: python run_dw_ptaL.py CASE   (CASE = ptaL [NG15, DW explains], ptaLE [EPTA, DW explains], ulL [NG15, upper limit])"""
import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
import fopt_dw_pipeline as fd
CASE = sys.argv[1] if len(sys.argv) > 1 else "ptaL"
d = dict(np.load("fopt_dw_samples.npz"))
mu, cov, acc, S, A = fd.gaussian_prior(fd.DW_BOX, lambda P: fd.dw_allowed(P, CASE), nsamp=1_000_000)
print(f"== DW[{CASE}]: allowed fraction {acc:.4f}; prior mean {np.round(mu, 3)}, std {np.round(np.sqrt(np.diag(cov)), 3)}")
d[f"DW_{CASE}_S"] = S
for Lt in (0.5, 0.8, 1.2):
    lM, lf, okd, detd = fd.dw_flags(S, Lt)
    d[f"DW_{CASE}_x_{Lt}"] = np.vstack([lM, lf, okd, detd])
    out = [fd.dw_flags_cc(S[i:i + 50000], Lt) for i in range(0, len(S), 50000)]
    lMs, lf2, ok, det, sD = (np.concatenate(z) for z in zip(*out))
    d[f"DW_{CASE}_cc_{Lt}"] = np.vstack([lMs, lf2, ok, det])
    sel = lf > -12
    print(f"   L/t={Lt}: P(f>1e-12)={np.mean(sel):.4f} P(f>1)={np.mean(lf > 0):.4f}  P(ok) mono {np.mean(sel & okd):.4f} / width {np.mean(sel & ok):.4f}"
          f"  P(ok&det) width {np.mean(sel & ok & det):.4f}  logM 5-50-95%: {np.round(np.percentile(lM[sel], [5, 50, 95]), 2) if sel.any() else None}")
np.savez("fopt_dw_samples.npz", **d)
