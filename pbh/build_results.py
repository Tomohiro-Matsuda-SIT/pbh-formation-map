"""
Build the per-(model, condition) result store (results_store.py).  usage: python build_results.py [model ...]
models: curvaton axion infcurv fopt dw fermiball infwall cstring reheat   (default: all)

Common conventions (2026-11-14)
  * GW constraints: LVK O3 pointwise; PTA = NANOGrav 15-yr HD free-spectrum likelihood, as an upper limit with
    SMBHB(13/3, amplitude profiled) + X (fopt_dw_pipeline.pta_ok_L), unless the condition says otherwise.
  * ok = passes current PBH bounds with the sample's own mass function (emf.py) AND limits on already-evaporated
    PBHs (evap.py).  det = exceeds a projected bound.
"""
import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
import results_store as rs
import emf, evap

MSUN_G = np.log10(1.98847e33)
ALL = ["curvaton", "axion", "infcurv", "fopt", "dw", "fermiball", "infwall", "cstring", "reheat"]
todo = sys.argv[1:] or ALL

def mono_okdet(lM, lf):
    """monochromatic: current/projected PBHbounds + evaporated-PBH limits for M < 5.1e14 g (f_eff = would-be f)."""
    cur = emf.mono_flags(lM, lf, emf.CURRENT); fut = emf.mono_flags(lM, lf, emf.FUTURE)
    ok = ~np.any(np.array(list(cur.values())), axis=0)
    det = np.any(np.array(list(fut.values())), axis=0)
    return ok, det

def mono_evap_ok(lM_g, lf_eff):
    fmax = np.min([evap.log_feff_max(k, lM_g) for k in evap.LIMITS], axis=0)
    return ~((lM_g < np.log10(5.1e14)) & (lf_eff > fmax))

# ------------------------------------------------------------------ curvaton family
if "curvaton" in todo:
    d = np.load("samples.npz")
    for zc in ("0.7", "1.0", "1.3"):
        lM, lf = d[f"lM_{zc}"], d[f"lf_{zc}"]
        ok, det = mono_okdet(lM, lf)
        rs.save("curvaton_std", f"zetac{zc}", lM, lf, ok, det,
                f"standard quadratic curvaton, threshold zeta_c = {zc}; Gaussian prior on CMB/classicality/BBN-allowed box",
                ["Starobinsky background fixed", "Sasaki-Valiviita-Wands nonlinear zeta", "monochromatic MF"],
                ["curvaton_pipeline.py"], theta=d["th"])
if "axion" in todo:
    d = np.load("axion_prior_samples.npz")
    for c in ("opt", "con"):
        lM, lf = d[f"lM_{c}"], d[f"lf_{c}"]
        ok, det = mono_okdet(lM, lf)
        rs.save("axion_curvaton", c, lM, lf, ok, det,
                f"axion-like (KKY) curvaton, hilltop treatment '{c}' ({'eps_min region counted' if c == 'opt' else 'eps_min region excluded'})",
                ["anharmonic G(theta) with hilltop asymptotics", "monochromatic MF"], ["axion_G.py", "axion_pipeline.py"],
                theta=d["P"])
if "infcurv" in todo:
    d = np.load("infcurv_scan.npz")
    for c in ("opt", "con"):
        lM, lf = d[f"lM_{c}"], d[f"lf_{c}"]
        ok, det = mono_okdet(lM, lf)
        rs.save("inflating_curvaton", c, lM, lf, ok, det,
                f"inflating curvaton: 16 optimisation points (maximal f), threshold variant '{c}'; NOT prior samples "
                "(no prior mass reaches the plane; global maximum by differential evolution: log10 f = -7.0e3 "
                "for M > 5e14 g, -3.8e3 over all masses; see infcurv_opt_de.py)", ["N2 <= 6.8 from n_s", "g2 >~ 0.23 (CMB amplitude with H_* = 1.6e13 GeV)"],
                ["infcurv_pipeline.py"], theta=d["P"])

# ------------------------------------------------------------------ FOPT (rerun with NG15 likelihood)
if "fopt" in todo:
    import fopt_dw_pipeline as fd
    fd.PTA_MODE = "L"
    mu, cov, acc, S, A = fd.gaussian_prior(fd.FOPT_BOX, fd.fopt_allowed)
    print(f"FOPT: allowed fraction {acc:.3f}, prior mean {np.round(mu, 3)}, std {np.round(np.sqrt(np.diag(cov)), 3)}")
    for dc in (0.4, 0.5, 0.667):
        out = [fd.fopt_flags(S[i:i + 50000], dc) for i in range(0, len(S), 50000)]
        lMs, lf, ok, det = (np.concatenate(z) for z in zip(*out))
        st = rs.save("fopt", f"GV_dc{dc}", lMs, lf, ok, det,
                     f"supercooled FOPT, non-covariant (Gouttenoire-Volansky), delta_c = {dc}; critical-collapse MF",
                     ["GW: envelope approx.; LVK O3 pointwise; PTA NG15 likelihood (upper limit with SMBHB)",
                      "lM = peak of critical-collapse MF", "evaporated-PBH limits applied"],
                     ["fopt_dw_pipeline.py", "emf.py", "evap.py", "pta_like.py"], theta=S)
        print(f"  GV dc={dc}: {st}")
        lM, lf = fd.fopt_map_cov(S, dc)
        ok, det = mono_okdet(lM, lf)
        st = rs.save("fopt", f"cov_dc{dc}", lM, lf, ok, det,
                     f"supercooled FOPT, covariant (Franciolini-Gouttenoire-Jinno), delta_c^(C) = {dc}",
                     ["FGJ P(delta) fit, Table I", "same prior as GV"], ["fopt_dw_pipeline.py"], theta=S, plot=True)
        print(f"  cov dc={dc}: P(f>1e-12)={st['P_f_gt_1e12']:.3g}")

# ------------------------------------------------------------------ DW (export existing prior samples)
if "dw" in todo:
    d = np.load("fopt_dw_samples.npz")
    desc = {"ulL": "PTA as upper limit: NG15 likelihood, SMBHB(13/3)+DW not worse than SMBHB alone by 1.92",
            "ptaL": "DW explains PTA: NG15 likelihood, DW-only dlnL >= -3 from best fit",
            "ptaLE": "DW explains PTA: EPTA DR2new likelihood (cross-check)",
            "ul": "LEGACY: PTA pointwise upper limit 1.51e-8 (f/fyr)^(2/3)",
            "pta": "LEGACY: DW explains PTA via Omega(1/yr) in [5.3e-9, 1.51e-8]"}
    for case in ("ulL", "ptaL", "ptaLE", "ul", "pta"):
        for Lt in ("0.5", "0.8", "1.2"):
            for mf, tag in (("cc", "width"), ("x", "mono")):
                lM, lf, ok, det = d[f"DW_{case}_{mf}_{Lt}"]
                rs.save("dw", f"{case}_Lt{Lt}_{tag}", lM, lf, ok, det,
                        f"biased domain walls (Gouttenoire-Vitagliano), L/t = {Lt}, MF = {tag}; {desc[case]}",
                        ["s_ball natural log (verified)", "GW: GV App. B.1",
                         "width: critical collapse with percolation-derived Delta scale (ours)" if tag == "width" else "monochromatic at t_dom/2G",
                         "evaporated-PBH limits applied (width only)" if tag == "width" else "PBHbounds only"],
                        ["fopt_dw_pipeline.py", "run_dw_ptaL.py", "dw_pta_fit.py", "pta_like.py"],
                        theta=d[f"DW_{case}_S"])
        print(f"  DW {case} done")

# ------------------------------------------------------------------ Fermi-ball (rerun with NG15 likelihood)
if "fermiball" in todo:
    import fopt_dw_pipeline as fd
    import fermiball_pipeline as fb
    fd.PTA_MODE = "L"
    mu, cov, acc, S, A = fd.gaussian_prior(fb.BOX, fb.fb_allowed)
    print(f"Fermi-ball: allowed fraction {acc:.3f}, prior mean {np.round(mu, 2)}")
    lM, lf = fb.fb_map(S)
    ok, det = mono_okdet(lM, lf)
    # evaporated samples: f_eff from the same formula without the M > 5.1e14 g cut
    lT, lbH, la, le = S.T
    import curvaton_pipeline as cp
    T, bH, a, eta = 10**lT, 10**lbH, 10**la, 10**le
    g = cp.gstar(T)
    Mg = 1.4e21 * fb.vb**3 * (eta / 1e-3) * (100 / g)**0.25 * (100 / T)**2 * (100 / bH)**3 * a**0.25
    feff = 1.3e3 * fb.vb**-3 * (g / 100)**0.5 * (T / 100)**3 * (bH / 100)**3 * (Mg / 1e15)
    ok &= mono_evap_ok(np.log10(Mg), np.log10(feff))
    st = rs.save("fermiball", "base", lM, lf, ok, det,
                 "Fermi-ball collapse (Kawana-Xie), v_b = 0.6, F_trap = 0.98; box log eta in [-16, -6]",
                 ["GW: sound waves with finite duration; LVK pointwise; PTA NG15 likelihood (upper limit with SMBHB)",
                  "untrapped-chi relic <= Omega_DM", "monochromatic MF", "evaporated-PBH limits applied"],
                 ["fermiball_pipeline.py", "fopt_dw_pipeline.py", "evap.py"], theta=S,
                 extra=dict(lM_all=np.log10(Mg) - MSUN_G, lf_eff=np.log10(feff)))
    print(f"  Fermi-ball: {st}")

# ------------------------------------------------------------------ inflationary walls, cosmic strings
if "infwall" in todo:
    d = np.load("infwall_samples.npz")
    par = {"opt": "A = 1e2, kappa = 2", "cen": "A = 1, kappa = 2", "con": "A = 1e-2, kappa = 1"}
    for c in ("opt", "cen", "con"):
        lM, lf, ok, det = d[c]
        rs.save("infwall", c, lM, lf, ok, det, f"walls nucleated during Starobinsky inflation (GVZ/DGV), {par[c]}",
                ["lambda ~ A exp(-2 pi^2 sigma / H^3)", "lM = peak of the non-evaporated MF",
                 "evaporated-PBH limits applied"], ["infwall_pipeline.py", "evap.py"])
if "cstring" in todo:
    d = np.load("cstring_samples.npz")
    for c, txt in (("free", "collapse fraction f free (log-Gaussian on [-30, 0])"),
                   ("CC", "collapse fraction from Caldwell-Casper fit, extrapolated")):
        lM, lf, ok, det = d[c]
        rs.save("cstring", c, lM, lf, ok, det, f"cosmic-string loop collapse (James-Turner et al. MF), {txt}",
                ["G mu <= min(T_max-bound, LVK 4e-15)", "lM = M_* = 5e14 g (abundance-weighted end)",
                 "evaporated-PBH limits (our evaluation; JT's 2e-12 not used)"], ["cosmicstring_pipeline.py", "evap.py"])
if "reheat" in todo:
    rs.save_note("starobinsky_reheat", "baseline",
                 "PBHs from the scalaron-dominated reheating era (MPV 1907.04236): horizon masses 13 g - 1.5e13 g, "
                 "all evaporated; nothing in the (M, f) plane. rho_Gamma^1/4 = 6.2e9 GeV vs MPV relic-overproduction "
                 "edge ~4e9 GeV (要確認)", ["Starobinsky background"], ["cosmicstring_pipeline.py (T_max)"])
print("done:", todo)
