"""
Per-(model, condition) result store.  Each result lives in its own directory

    results/<model>/<condition>/
        samples.npz   lM [log10 M/Msun, representative mass], lf [log10 f_PBH today], ok [passes current bounds
                      with its own mass function, incl. evaporated-PBH limits], det [detectable by projected probes],
                      optional theta (prior samples), and any extra arrays
        meta.json     model, condition, description, assumptions, prior, code, date, summary statistics
        map.pdf/png   single-panel (M, f_PBH) figure of this result alone: pushforward probability per 1/3-dex cell
                      (samples with ok = True), reach contour p >= EPS, current / projected bound envelopes

results/index.json lists all stored results.  The combined (A)/(B) figure (fig_combined.py) only reads this store.
Grid, envelopes (PBHbounds commit 96bc091) and EPS are defined here once and shared.
"""
from paths import PBH_ROOT, PBHBOUNDS, CEFFYL, NG15, NPROC
import os, json, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

ROOT = PBH_ROOT + "/results"
_RESERVED = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(10)} | {f"lpt{i}" for i in range(10)}

def _dir(model, cond):
    """directory of a stored result.  Windows forbids reserved device names (e.g. 'con') as path segments,
    so those are stored as '_con' on disk; the condition key itself (index.json, meta.json) is unchanged."""
    seg = lambda s: "_" + s if s.split(".")[0].lower() in _RESERVED else s
    return f"{ROOT}/{seg(model)}/{seg(cond)}"
PB = PBHBOUNDS + "/bounds/"
EPS = 1e-5
xe = np.arange(-18, 6 + 1e-9, 1 / 3); ye = np.arange(-12, 0 + 1e-9, 1 / 3)
xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
CURRENT = ["Evaporation", "Microlensing", "GWs", "Accretion", "Dynamical"]
FUTURE = ["3G-GW-10y", "GECCO", "Xraylensing-Xmu-proj", "Xraylensing-eXTP-proj", "WhiteDwarfmicro",
          "PTA-Doppler", "PTA-Shapiro", "SKA", "GRB-parallax", "FL-small"]
MEVAP = np.log10(5e14 / 1.98847e33)
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True})

def _load(name):
    d = np.loadtxt(PB + name + ".txt", comments="#", usecols=(0, 1))
    d = d[np.isfinite(d).all(1) & (d[:, 0] > 0) & (d[:, 1] > 0)]
    return np.log10(d[:, 0]), np.log10(d[:, 1])

def envelope(names):
    env = np.full(len(xc), np.inf)
    for nm in names:
        lm, lf = _load(nm)
        o = np.argsort(lm); lm, lf = lm[o], lf[o]
        inside = (xc >= lm.min()) & (xc <= lm.max())
        env[inside] = np.minimum(env[inside], np.interp(xc[inside], lm, lf))
    return env

ENV_CUR = envelope(CURRENT)
ENV_FUT = np.minimum(ENV_CUR, envelope(FUTURE))

def pmap(lM, lf, w=None):
    m = np.isfinite(lM) & np.isfinite(lf)
    ww = None if w is None else np.asarray(w)[m].astype(float)
    H, _, _ = np.histogram2d(lM[m], lf[m], bins=[xe, ye], weights=ww)
    return H / max(len(lM), 1)

def maps(res):
    """-> (p_reach, p_reach_detectable) on the grid (ok-weighted).  If the result carries prior weights 'w'
    (importance / conditional integration, e.g. the USR channel), cell probabilities are sums of w."""
    ok = res["ok"].astype(bool); det = res["det"].astype(bool)
    if "w" in res:
        w = np.asarray(res["w"], float)
        m = np.isfinite(res["lM"]) & np.isfinite(res["lf"])
        P, _, _ = np.histogram2d(res["lM"][m], res["lf"][m], bins=[xe, ye], weights=(w * ok)[m])
        Pd, _, _ = np.histogram2d(res["lM"][m], res["lf"][m], bins=[xe, ye], weights=(w * (ok & det))[m])
        return P, Pd
    return pmap(res["lM"], res["lf"], ok), pmap(res["lM"], res["lf"], ok & det)

def stats(lM, lf, ok, det, w=None):
    pb = lf > -12; okb = ok.astype(bool); dt = det.astype(bool)
    inwin = pb & (lM >= xe[0]) & (lM <= xe[-1])
    W = np.full(len(lf), 1 / max(len(lf), 1)) if w is None else np.asarray(w, float)
    out = dict(n=int(len(lf)), P_f_gt_1e12=float(np.sum(W * pb)), P_f_gt_1=float(np.sum(W * (lf > 0))),
               P_reach=float(np.sum(W * (pb & okb))), P_reach_det=float(np.sum(W * (pb & okb & dt))),
               P_reach_in_window=float(np.sum(W * (inwin & okb))), weighted=w is not None)
    if (pb & okb).any():
        sel = pb & okb
        o = np.argsort(lM[sel]); cw = np.cumsum(W[sel][o]); cw /= cw[-1]
        out["logM_5_50_95_reach"] = [float(np.interp(q, cw, lM[sel][o])) for q in (0.05, 0.5, 0.95)]
    return out

XW = np.arange(-22, 8 + 1e-9, 1 / 3)            # wider mass window for the individual figures
XWc = 0.5 * (XW[1:] + XW[:-1])

def _plot(path, res, title, note):
    """individual figure: wider mass window (shows samples in the evaporated range too)."""
    lM, lf, ok = res["lM"], res["lf"], res["ok"].astype(bool)
    m = np.isfinite(lM) & np.isfinite(lf)
    if "w" in res:
        P, _, _ = np.histogram2d(lM[m], lf[m], bins=[XW, ye], weights=(np.asarray(res["w"], float) * ok)[m])
    else:
        P, _, _ = np.histogram2d(lM[m], lf[m], bins=[XW, ye], weights=ok[m].astype(float))
        P /= max(len(lM), 1)
    xe_, xc_ = XW, XWc
    env_c = np.interp(xc_, xc, ENV_CUR, left=np.inf, right=np.inf)
    env_f = np.interp(xc_, xc, ENV_FUT, left=np.inf, right=np.inf)
    fig, ax = plt.subplots(figsize=(4.6, 3.6), layout="constrained")
    Pm = np.ma.masked_less_equal(P, 0)
    if Pm.count():
        vmax = Pm.max(); vmin = max(Pm.min(), vmax * 1e-4)
        pc = ax.pcolormesh(xe_, ye, Pm.T, cmap="Blues", norm=LogNorm(vmin=vmin, vmax=vmax),
                           edgecolors="face", linewidth=0, rasterized=True)
        fig.colorbar(pc, ax=ax, label="prior probability per cell (allowed samples)")
        if (P >= EPS).any():
            ax.contour(xc_, yc, (P >= EPS).astype(float).T, levels=[0.5], colors="k", linewidths=1.0)
    ax.fill_between(xc, np.clip(ENV_CUR, -12, 0), 0, color="#d9d9d9", step="mid", zorder=0)
    ax.axvspan(xe_[0], MEVAP, color="#eeeeee", zorder=0)
    ax.text(0.5 * (xe_[0] + MEVAP), -1, "evaporated\n(today)", ha="center", fontsize=6.5, color="0.4")
    ax.axvline(xe[0], color="0.5", lw=0.6, ls=":")
    ax.plot(xc, np.clip(ENV_CUR, -12, 0.5), color="0.25", lw=0.8)
    ax.plot(xc, np.clip(ENV_FUT, -12, 0.5), color="0.25", lw=0.8, ls=(0, (2, 2)))
    ax.set_xlim(xe_[0], xe_[-1]); ax.set_ylim(ye[0], ye[-1])
    ax.set_xlabel(r"$\log_{10}(M/M_\odot)$"); ax.set_ylabel(r"$\log_{10} f_{\rm PBH}$")
    ax.set_title(title, fontsize=8.5, loc="left")
    if Pm.count():
        note = note + f"\nmax p per cell = {Pm.max():.2g}"
    ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=6.0, va="bottom", wrap=True)
    ax.legend(handles=[Line2D([], [], color="k", lw=1, label=r"reach $p\geq\epsilon=10^{-5}$"),
                       Line2D([], [], color="0.25", lw=0.8, label="current bounds"),
                       Line2D([], [], color="0.25", lw=0.8, ls=(0, (2, 2)), label="projected")],
              fontsize=6.5, loc="upper right", frameon=False)
    ax.text(0.98, 0.70, "dotted: left edge of the combined figure", transform=ax.transAxes, fontsize=5.5,
            ha="right", color="0.4")
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}/map.{ext}", dpi=200)
    plt.close(fig)

def save(model, cond, lM, lf, ok, det, description, assumptions=None, code=None, theta=None, extra=None, plot=True):
    path = _dir(model, cond)
    os.makedirs(path, exist_ok=True)
    lM = np.asarray(lM, float); lf = np.asarray(lf, float)
    ok = np.asarray(ok).astype(bool); det = np.asarray(det).astype(bool)
    arrs = dict(lM=lM, lf=lf, ok=ok, det=det)
    if theta is not None:
        arrs["theta"] = theta
    if extra:
        arrs.update(extra)
    np.savez_compressed(f"{path}/samples.npz", **arrs)
    st = stats(lM, lf, ok, det, arrs.get("w"))
    meta = dict(model=model, condition=cond, description=description, assumptions=assumptions or [],
                code=code or [], date=datetime.date.today().isoformat(), eps=EPS, stats=st)
    json.dump(meta, open(f"{path}/meta.json", "w"), ensure_ascii=False, indent=1)
    if plot:
        note = (f"P(f>1e-12)={st['P_f_gt_1e12']:.3g}  P(reach)={st['P_reach']:.3g}\n"
                f"P(reach&det)={st['P_reach_det']:.3g}")
        _plot(path, arrs, f"{model} / {cond}", note)
    idx_path = f"{ROOT}/index.json"
    idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
    idx[f"{model}/{cond}"] = dict(description=description, date=meta["date"], stats=st)
    json.dump(dict(sorted(idx.items())), open(idx_path, "w"), ensure_ascii=False, indent=1)
    return st

def save_note(model, cond, description, assumptions=None, code=None):
    """for mechanisms with no samples in the plane (e.g. Starobinsky preheating baseline)."""
    path = _dir(model, cond)
    os.makedirs(path, exist_ok=True)
    meta = dict(model=model, condition=cond, description=description, assumptions=assumptions or [],
                code=code or [], date=datetime.date.today().isoformat(), stats=None)
    json.dump(meta, open(f"{path}/meta.json", "w"), ensure_ascii=False, indent=1)
    idx_path = f"{ROOT}/index.json"
    idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
    idx[f"{model}/{cond}"] = dict(description=description, date=meta["date"], stats=None)
    json.dump(dict(sorted(idx.items())), open(idx_path, "w"), ensure_ascii=False, indent=1)

def load(model, cond):
    d = np.load(f"{_dir(model, cond)}/samples.npz")
    return {k: d[k] for k in d}

def replot_all():
    """regenerate every map.pdf/png from stored samples (no recomputation)."""
    import glob
    for p in sorted(glob.glob(f"{ROOT}/*/*/samples.npz")):
        path = os.path.dirname(p)
        meta = json.load(open(f"{path}/meta.json"))
        res = dict(np.load(p))
        st = meta["stats"]
        if st is None:
            continue
        note = (f"P(f>1e-12)={st['P_f_gt_1e12']:.3g}  P(reach)={st['P_reach']:.3g}\n"
                f"P(reach&det)={st['P_reach_det']:.3g}")
        _plot(path, res, f"{meta['model']} / {meta['condition']}", note)

if __name__ == "__main__":
    replot_all()
