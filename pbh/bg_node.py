r"""
One background node of the hierarchical-prior rerun (bg_run.py launches it with PBH_BG set, so that
curvaton_pipeline.BG and every model module see the alpha-attractor background of this node).
usage: python bg_node.py OUT.npz NSAMP

For each model the per-model prior is rebuilt ON THIS BACKGROUND with the project rule (Gaussian on the
allowed region of the box; box edges that depend on the background -- T < T_reh, sigma^{1/3} < T_reh,
sigma^{1/3}/H_* <= M_P/H_*, G mu <= G mu(T_max) -- move with it).  Stored per condition: lM, lf, ok, det
(float32 / bool) plus the per-node prior summary.
Models rerun here (background enters through T_reh, H(N), H_*, T_max):
  FOPT (GV, delta_c band; covariant FGJ), DW (ulL / ptaL, critical-collapse width, L/t band),
  Fermi-ball, inflationary walls (A, kappa band), cosmic strings (free / CC).
NOT rerun (stay on the fixed Starobinsky background, see docs): curvaton family (their Omega formulas assume
MD reheating, w = 0), inflating curvaton (Starobinsky-specific N_2 construction), USR bump (own background).
"""
import os, sys, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
import curvaton_pipeline as cp
import fopt_dw_pipeline as fd
import emf, evap

OUT = sys.argv[1]; NS = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
# robustness switches (robustness.py): prior width (covariance x PSCALE^2, centre kept) and box-edge variants
PSCALE = float(os.environ.get("PBH_PSCALE", "1"))
BOXVAR = os.environ.get("PBH_BOXVAR", "base")
MEASURE = os.environ.get("PBH_MEASURE", "gauss")     # gauss (project rule) | flat (uniform over the allowed region)
EDGE = {  # (base, narrow, wide)
    "fopt_bH_hi": dict(base=30.0, narrow=15.0, wide=100.0),     # beta/H upper edge
    "dw_la_lo": dict(base=-4.0, narrow=-3.0, wide=-5.0),         # log10 alpha_ann lower edge
    "fb_eta_lo": dict(base=-16.0, narrow=-14.0, wide=-18.0),     # log10 eta_chi lower edge
    "iw_hi": dict(base=1.0, narrow=0.1, wide=1.0),               # sigma^{1/3} upper edge in units of M_P
    "cs_lg_lo": dict(base=-26.0, narrow=-22.0, wide=-30.0),      # log10 G mu lower edge
}
E = {k: v[BOXVAR] for k, v in EDGE.items()}
import zlib
seed = zlib.crc32(os.path.basename(OUT).encode())
fd.rng = np.random.default_rng(seed)
fd.PTA_MODE = "L"
res, info = {}, {"bg": {k: v for k, v in cp.BG.items()}, "nsamp": NS}
t0 = time.time()

def put(tag, lM, lf, ok, det):
    res[tag] = np.vstack([np.asarray(lM, np.float32), np.asarray(lf, np.float32),
                          np.asarray(ok, np.float32), np.asarray(det, np.float32)])

def mono_okdet(lM, lf):
    cur = emf.mono_flags(lM, lf, emf.CURRENT); fut = emf.mono_flags(lM, lf, emf.FUTURE)
    return ~np.any(np.array(list(cur.values())), axis=0), np.any(np.array(list(fut.values())), axis=0)

def prior(box, fn, nbox=200_000, nmin=30):
    """fd.gaussian_prior with a guard against an (almost) empty allowed region on this background."""
    P = box[:, 0] + (box[:, 1] - box[:, 0]) * fd.rng.random((nbox, len(box)))
    A = P[fn(P)]
    if len(A) < nmin:
        raise ValueError(f"allowed region too small ({len(A)} of {nbox})")
    mu, cov = A.mean(0), np.cov(A.T) * PSCALE**2
    out, tot = [], 0
    # MEASURE = flat: uniform over the allowed region R in the box coordinates (drawn inside the bounding box of
    # the allowed points, padded by 25% of its extent on each side, to keep the acceptance usable)
    lo_, hi_ = A.min(0), A.max(0); pad = 0.25 * (hi_ - lo_)
    lo_, hi_ = np.maximum(lo_ - pad, box[:, 0]), np.minimum(hi_ + pad, box[:, 1])
    for _ in range(2000):
        if MEASURE == "flat":
            Q = lo_ + (hi_ - lo_) * fd.rng.random((2 * NS, len(box)))
        else:
            Q = fd.rng.multivariate_normal(mu, cov, size=2 * NS)
        Q = Q[np.all((Q >= box[:, 0]) & (Q <= box[:, 1]), axis=1)]
        Q = Q[fn(Q)]
        out.append(Q); tot += len(Q)
        if tot >= NS:
            break
    return mu, cov, len(A) / nbox, np.concatenate(out)[:NS]

# ---------------- FOPT ----------------
box = np.array([[np.log10(5e-3), np.log10(cp.BG["Treh"])], [3.0, E["fopt_bH_hi"]]])
try:
    mu, cov, acc, S = prior(box, fd.fopt_allowed)
except ValueError as e:
    info["fopt"] = dict(acc=0.0, err=str(e)); S = None
if S is not None:
    info["fopt"] = dict(acc=acc, mean=mu.tolist())
    for dc in (0.4, 0.5, 0.667):
        out = [fd.fopt_flags(S[i:i + 50000], dc) for i in range(0, len(S), 50000)]
        put(f"fopt/GV_dc{dc}", *(np.concatenate(z) for z in zip(*out)))
        lM, lf = fd.fopt_map_cov(S, dc); put(f"fopt/cov_dc{dc}", lM, lf, *mono_okdet(lM, lf))
info["t_fopt"] = time.time() - t0

# ---------------- DW ----------------
dwbox = fd.DW_BOX.copy(); dwbox[1, 0] = E["dw_la_lo"]
for case in ("ulL", "ptaL"):
    try:
        mu, cov, acc, S = prior(dwbox, lambda P: fd.dw_allowed(P, case))
    except Exception as e:                     # empty allowed region (e.g. T_reh below the PTA-band T_ann)
        info[f"dw_{case}"] = dict(acc=0.0, err=str(e)); continue
    info[f"dw_{case}"] = dict(acc=acc, mean=mu.tolist())
    for Lt in (0.5, 0.8, 1.2):
        out = [fd.dw_flags_cc(S[i:i + 50000], Lt) for i in range(0, len(S), 50000)]
        lMs, lf, ok, det, _ = (np.concatenate(z) for z in zip(*out))
        put(f"dw/{case}_Lt{Lt}_width", lMs, lf, ok, det)
info["t_dw"] = time.time() - t0

# ---------------- Fermi-ball ----------------
import fermiball_pipeline as fb
fb.TREH = cp.BG["Treh"]
fbbox = fb.BOX.copy(); fbbox[0, 1] = np.log10(cp.BG["Treh"]); fbbox[3, 0] = E["fb_eta_lo"]
try:
    mu, cov, acc, S = prior(fbbox, fb.fb_allowed)
except ValueError as e:
    info["fermiball"] = dict(acc=0.0, err=str(e)); S = None
if S is not None:
    info["fermiball"] = dict(acc=acc, mean=mu.tolist())
    lM, lf = fb.fb_map(S)
    ok, det = mono_okdet(lM, lf)
    lT, lbH, la, le = S.T
    T, bH, a, eta = 10**lT, 10**lbH, 10**la, 10**le
    g = cp.gstar(T)
    Mg = 1.4e21 * fb.vb**3 * (eta / 1e-3) * (100 / g)**0.25 * (100 / T)**2 * (100 / bH)**3 * a**0.25
    feff = 1.3e3 * fb.vb**-3 * (g / 100)**0.5 * (T / 100)**3 * (bH / 100)**3 * (Mg / 1e15)
    fmax = np.min([evap.log_feff_max(k, np.log10(Mg)) for k in evap.LIMITS], axis=0)
    ok &= ~((np.log10(Mg) < np.log10(5.1e14)) & (np.log10(feff) > fmax))
    put("fermiball/base", lM, lf, ok, det)
info["t_fb"] = time.time() - t0

# ---------------- inflationary walls ----------------
import infwall_pipeline as iw
iw.rng = np.random.default_rng(seed + 1)
nw = max(NS // 5, 5000)

def iw_run(A, kap, n):
    """infwall_pipeline.run with the robustness switches (upper edge sigma^{1/3} <= E['iw_hi'] M_P, width x PSCALE)."""
    b = iw.box(A); b[1] = np.log10(E["iw_hi"] * iw.MP / cp.BG["Hst"])
    u = b[0] + (b[1] - b[0]) * iw.rng.random(400_000)
    x = iw.rng.normal(u.mean(), u.std() * PSCALE, 6 * n); x = x[(x >= b[0]) & (x <= b[1])][:n]
    if MEASURE == "flat":
        x = u[:n]
    out = []
    for i in range(0, len(x), 1500):
        lM, psi, ftot, lMpk = iw.wall_mf(x[i:i + 1500], A, kap)
        logf = np.where(ftot > 0, np.log10(np.maximum(ftot, 1e-300)), -np.inf)
        sel = logf > -12
        ok = iw.wall_mf.evap_ok.copy(); det = np.zeros(len(logf), bool)
        if sel.any():
            Ic = emf.integrals(lM[sel], psi[sel], emf.CURRENT); If = emf.integrals(lM[sel], psi[sel], emf.FUTURE)
            ok[sel] &= np.all(np.array([Ic[k] <= 1 for k in Ic]), axis=0)
            det[sel] = np.any(np.array([If[k] > 1 for k in If]), axis=0)
        out.append((lMpk, logf, ok, det))
    return (np.concatenate(z) for z in zip(*out))

for tag, A, kap in (("opt", 1e2, 2.0), ("cen", 1.0, 2.0), ("con", 1e-2, 1.0)):
    if not iw.keep.any():
        break
    put(f"infwall/{tag}", *iw_run(A, kap, nw))
info["t_iw"] = time.time() - t0

# ---------------- cosmic strings ----------------
import cosmicstring_pipeline as cs
Tm, _ = cs.T_max()
lg_hi = np.log10(min(cs.gmu_of_eta(Tm), 4e-15)); lg_lo = E["cs_lg_lo"]
info["cstring"] = dict(Tmax=Tm, lg_hi=lg_hi)
r = np.random.default_rng(seed + 2); nc = max(NS // 50, 1000)
for tag in ("CC", "free"):
    mu_, sd_ = 0.5 * (lg_lo + lg_hi), (lg_hi - lg_lo) / np.sqrt(12) * PSCALE
    lg = r.normal(mu_, sd_, 8 * nc); lg = lg[(lg > lg_lo) & (lg < lg_hi)][:nc]
    if MEASURE == "flat":
        lg = r.uniform(lg_lo, lg_hi, nc)
    if tag == "CC":
        lf = cs.f_CC(lg)
    else:
        lf = r.normal(-15, 30 / np.sqrt(12) * PSCALE, 8 * nc); lf = lf[(lf > -30) & (lf < 0)][:len(lg)]
        if MEASURE == "flat":
            lf = r.uniform(-30, 0, len(lg))
        lg = lg[:len(lf)]
    lc = np.log10(cs.c_string(lf, lg))
    rr = np.array([cs.flags(a, b) for a, b in zip(lc, lg)])
    ftot, ok, det = rr.T
    put(f"cstring/{tag}", np.full(len(lg), np.log10(cs.MSTAR / cs.MSUN)), np.log10(np.maximum(ftot, 1e-300)), ok, det)
info["t_total"] = time.time() - t0
np.savez_compressed(OUT, info=json.dumps(info, default=float), **{k.replace("/", "__"): v for k, v in res.items()})
print(f"{os.path.basename(OUT)}: T_reh={cp.BG['Treh']:.3g} w={cp.BG['w']:.2f} alpha={cp.BG['alpha']:.3g} "
      f"N*={cp.BG['Nst']:.1f}  done in {info['t_total']:.0f}s")
