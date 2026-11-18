# AI guide to the PBH reachability analysis

This file is written to be loaded into an AI assistant, together with `docs/analysis_notes.pdf` (or its source `docs/analysis_notes.tex`). It explains the analysis of the paper

> T. Matsuda, *Where primordial black holes can come from: how likely formation mechanisms are to produce a given mass and abundance*

in a form that an assistant can follow step by step. With it, you can do three things:

- reproduce the paper's numbers;
- change one ingredient;
- repeat the whole analysis from the beginning with formation models of your own.

Human readers can use it as a technical README.

**How to use it with an assistant.** Give the assistant this file and the analysis notes, then state the task. For example:

- "Read AI_GUIDE.md and analysis_notes.tex. Explain how the probability per square is computed for domain walls, and point to the lines of code."
- "Following Section 6 of AI_GUIDE.md, add a new formation mechanism with parameters (a, b, c) and the mass and abundance formulas below. Then recompute Fig. 1 of the paper."
- "Using only the interface in Section 5, rebuild the analysis from scratch for my own set of models in a new repository."

Ask the assistant to state its assumptions and to check its numbers against Section 8.

---

## 1. What the analysis computes, in one paragraph

A formation mechanism has parameters θ. These parameters fix two things:

- the PBH mass function ψ(M; θ) = d f_PBH / d ln M;
- other observables: the CMB, gravitational-wave (GW) backgrounds, and BBN.

For each mechanism we first find the region R of θ that is allowed by the **non-PBH** observations only. We then assign θ a probability distribution π(θ): a Gaussian with the mean and covariance of a uniform distribution over R, set to zero outside R.

Each sample of θ is turned into two numbers:

- the peak mass M_* of its mass function;
- its present abundance f_PBH.

It is also flagged as allowed or not by the **current PBH limits**, applied to its own mass function and including limits on PBHs that have already evaporated.

The plane log10(M/Msun) ∈ [−18, 6], log10 f_PBH ∈ [−12, 0] is divided into squares 1/3 decade wide. For every square c, p_c is the probability that the mechanism produces allowed PBHs in c (Eq. (1) of the paper). The main result is the map of max over mechanisms of p_c (Fig. 1 of the paper).

As a summary, a square is "reached" if p_c ≥ ε, with ε = 1e-5 per square. This value equals 9e-5 per decade² and is held fixed per unit area when the square size changes. Reached squares are then sorted into classes:

- **(A)**: allowed by observations, but not reached;
- **(B1)**: reached, and at least half of the reaching probability lies above a planned limit, so it is testable;
- **(B2)**: reached, but beyond planned sensitivity.

Every choice that enters is varied:

- the threshold;
- the form of the distribution (Gaussian or uniform);
- its spread;
- the theoretical range limits;
- the square size;
- the inflation model, including a family with free reheating temperature and its members that agree with ACT.

---

## 2. Glossary (exact meanings used in code and paper)

| Term | Meaning |
|---|---|
| mechanism (code: `model`) | a formation mechanism, e.g. `fopt`, `dw`, `fermiball` |
| treatment (code: `condition`) | one theoretical variant of a mechanism, e.g. `GV_dc0.5` (non-covariant FOPT, δ_c = 0.5) |
| box B | the parameter ranges in which R is searched (`BOX` arrays in each module) |
| allowed region R | the part of B that satisfies the non-PBH conditions (`*_allowed(P)` / `constraints(P)` functions) |
| parameter distribution | truncated Gaussian on R with the moments of uniform-on-R; the paper also calls it "prior" once |
| sample | one draw θ_i, stored as `lM` (log10 M_*/Msun), `lf` (log10 f_PBH today), `ok`, `det`, optional weight `w` |
| `ok` | passes **all current** PBH limits with its own mass function, **including limits on evaporated PBHs** |
| `det` | exceeds at least one **planned** limit |
| p_c | Σ over samples in square c of weight × ok (weight 1/N for unweighted results) |
| P_reach | Σ over samples of weight × ok × (f_PBH > 1e-12): the probability of producing allowed PBHs |
| scenario | one panel of Figs. 1–4 of the paper: (theory column: optimistic / conservative) × (PTA interpretation: upper limit / domain walls produce the signal) |
| reference inflation model | Starobinsky inflation with scalaron→Higgs reheating, T_reh = 2.5e9 GeV, N_* = 51.0 |

---

## 3. Repository layout

```
README.md            human-oriented overview
AI_GUIDE.md          this file
requirements.txt     numpy>=2.0, scipy, matplotlib
setup.ps1 / setup.sh environment + data check
docs/analysis_notes.tex/.pdf   formulas, ranges, constraints, statistics, figure construction, analytic checks
external/            PBHbounds (commit 96bc091), ceffyl data (commit 0cd2b3a), NANOGrav 15-yr KDE (Zenodo 10344086)
expected/            reference values used by regenerate.py for automatic comparison
pbh/                 all code; results are written here
  paths.py           every file location (override with env vars PBH_ROOT, PBHBOUNDS, CEFFYL, NG15; PBH_NPROC)
  results/<model>/<condition>/{samples.npz, meta.json, map.pdf}   the result store
```

Table I of the analysis notes maps every part of the analysis to its file. In short:

- **Shared machinery**:
  - `results_store.py`: grid, limit curves, ε, store;
  - `emf.py`: limits on extended mass functions;
  - `evap.py`: limits on evaporated PBHs;
  - `pta_like.py`: NANOGrav likelihood;
  - `curvaton_pipeline.py`: reference background and conversion tables;
  - `gstar_ss.py`: g_ρ and g_s.
- **One module per mechanism**:
  - `curvaton_pipeline.py`, `axion_pipeline.py`, `infcurv_pipeline.py`;
  - `fopt_dw_pipeline.py`, `fermiball_pipeline.py`;
  - `infwall_pipeline.py`, `cosmicstring_pipeline.py`;
  - `usr_*.py` (inflaton bump).
- **Assembly**: `build_results.py` writes the store. `fig_combined.py` defines the scenarios and the classes.
- **Variations and maps**:
  - `measure_checks.py`, `measure_plots.py`, `robustness.py`;
  - `bg_*.py` (inflation-model family);
  - `bg_prior_act.py`, `bg_act_combine.py` (ACT-compatible members).
- **Checks**: `analytic.py`, `infcurv_opt_de.py`, `axion_target_de.py`, `fopt_gwvar.py`, `dw_bayes.py`, `infwall_reh.py`, `axion_zc.py`.

---

## 4. Data flow

```
non-PBH conditions ──► allowed region R ──► parameter distribution π(θ) ──► samples θ_i
                                                                            │
mechanism map θ ─► ψ(M;θ) ─► (M_*, f_PBH) ; ok (current + evaporated limits) ; det (planned limits)
                                                                            │
                                            results/<model>/<condition>/samples.npz   (results_store.save)
                                                                            │
     fig_combined.SCEN: which conditions form each scenario ──► per square p_c, max over treatments,
                                                                 max over mechanisms ──► maps, classes
```

Rules that keep the analysis consistent. Keep them when you extend it.

1. **PBH limits never enter R.** R uses only CMB, GW (LVK, NANOGrav), BBN and thermal-history conditions. PBH limits enter only through `ok`.
2. **Figures never recompute a mechanism.** They read the store only.
3. Within a mechanism and scenario, treatments combine by **maximum** of p_c. Across mechanisms they combine by **maximum** for the map and by **union** for "reached".
4. A sample contributes to a square only if `ok` is true and f_PBH > 1e-12 (the lower edge of the plane).
5. Squares above the strongest current limit, or with M < 5e14 g (evaporated today), are excluded from the plane.

---

## 5. The interface a mechanism must provide

A mechanism is defined by four things, each implemented as plain numpy functions on arrays of parameter vectors `P` of shape (n, d):

| Item | Signature (convention) | Notes |
|---|---|---|
| box | `BOX = np.array([[lo, hi], ...])` | logarithmic parameters where natural; mark which edges are theory choices |
| allowed region | `allowed(P) -> bool[n]` | non-PBH conditions only |
| map | `pbh_map(P) -> (lM, lf)` and, if the mass function is extended, the grid `logM (n,m)` and `psi (n,m)` = d f/d ln M | `lM` = log10 of the peak mass in Msun; `lf` = log10 f_PBH today, −inf if none |
| flags | `ok, det` | see below |

**Flags.**

- *Monochromatic mass function:*
  - `cur = emf.mono_flags(lM, lf, emf.CURRENT)`; `ok = ~any(cur.values())`
  - `fut = emf.mono_flags(lM, lf, emf.FUTURE)`; `det = any(fut.values())`
  - Then add the limits on evaporated PBHs for M < 5.1e14 g. See `build_results.mono_evap_ok` and `evap.log_feff_max`.
- *Extended mass function:* use `emf.integrals(logM, psi, emf.CURRENT)` for the part of ψ that survives today, and require every integral to be ≤ 1. Apply `evap.integrals(logM_g, psi_eff)` to the evaporated part, where `psi_eff` is the abundance it would have today had it not evaporated. See `fopt_dw_pipeline.fopt_flags` and `fopt_dw_pipeline.dw_flags_cc` for complete examples.

**Distribution and storage.**

```python
import numpy as np, results_store as rs, fopt_dw_pipeline as fd
mu, cov, acc, S, A = fd.gaussian_prior(BOX, allowed, nbox=400_000, nsamp=1_000_000)
lM, lf = pbh_map(S)                      # plus psi if extended
ok, det = flags(...)                     # as above
rs.save("mymech", "base", lM, lf, ok, det,
        description="what this treatment is",
        assumptions=["...", "..."], code=["mymech_pipeline.py"], theta=S)
```

`fd.gaussian_prior` draws points uniformly in the box, keeps the allowed ones, and takes their mean and covariance. It then samples the Gaussian, keeping draws inside box ∩ R. The rerun variations use the same rule in `bg_node.prior` with switches:

- `PBH_PSCALE`: spread;
- `PBH_MEASURE=flat`: uniform on R;
- `PBH_BOXVAR`: range limits.

If the probability is concentrated in a set far too thin to sample, use weighted samples instead. An example is the inflaton bump, where f_PBH goes from 1e-12 to 1 within Δ ln A ~ 1e-5. Integrate the distribution across the set directly and pass `extra=dict(w=weights)` to `rs.save`. The weights must sum to the probability they represent. `rs.maps` uses them automatically. See `usr_pushforward.py`.

**Adding the mechanism to the combined results.** Add an entry to `SCEN` in `fig_combined.py`, for each scenario in which it belongs. List all treatments whose maximum should be taken. `measure_checks.py` and `robustness.py` read `SCEN`. If the mechanism depends on the inflation model (through T_reh, H(N) or T_max), also add it to `bg_node.py` so that the variations and the inflation-model family include it.

---

## 6. Recipes

### 6.1 Reproduce the paper
1. Run `setup.ps1` or `setup.sh`. Put the external data in `external/` (see `paths.py`).
2. `cd pbh && python regenerate.py`. This rebuilds the store and compares it with `expected/`. It takes 1.5–2.5 h on 8 cores.
3. Run the additional steps in Sec. X of the analysis notes, in order:
   - fine grid of bump positions;
   - inflation-model family;
   - variations;
   - ACT models;
   - maps;
   - checks;
   - figures.

### 6.2 Add a mechanism to this analysis
1. Write `pbh/mymech_pipeline.py` with the four items of Section 5. Put a header listing every assumption with references, as the existing modules do.
2. Check that a few hand-picked θ give the published (M, f_PBH) of the original paper of the mechanism.
3. Store the treatments with `rs.save`. Inspect `results/mymech/<condition>/map.pdf`.
4. Add the mechanism to `SCEN` and, if needed, to `bg_node.py`.
5. Run `python fig_combined.py`, `python measure_checks.py`, and `python measure_plots.py`.
6. Report how max p_c, the (A) fraction as a function of ε, and the squares without samples change.

### 6.3 Start from scratch with different models
You can keep the shared machinery and replace all mechanisms:

- `results_store.py`, `emf.py`, `evap.py`, `pta_like.py`, `gstar_ss.py`;
- the background modules, if your mechanisms live on the same inflation models.

The minimum to reimplement for each new mechanism is Section 5. If you change the plane, the square size or ε, edit `results_store.py` (`xe`, `ye`, `EPS`). If you change the PBH limits, edit `results_store.CURRENT`/`FUTURE` and `emf.CURRENT`/`FUTURE`. Keep rule 1 of Section 4: the distribution must not use PBH data. Otherwise the comparison with the PBH limits is circular.

### 6.4 Change one ingredient
| Ingredient | Where |
|---|---|
| threshold ε, grid | `results_store.py` (`EPS`, `xe`, `ye`); threshold curves need no rerun (`measure_checks.py`) |
| form/spread of the distribution, range limits | `bg_node.py` switches `PBH_MEASURE`, `PBH_PSCALE`, `PBH_BOXVAR`; `robustness.py run` |
| PTA interpretation | `fopt_dw_pipeline.dw_allowed(P, case)`; likelihood in `pta_like.py` |
| FOPT GW spectrum | env var `PBH_FOPT_GW` = `env` / `LV3` / `LV2` (`fopt_dw_pipeline.py`) |
| inflation model | `PBH_BG` = path of a background JSON written by `background.write_bg`; families via `bg_prior.py` / `bg_prior_act.py` and `bg_run.py` |

---

## 7. Conventions and pitfalls

- **Units.**
  - Masses in the plane are log10(M/Msun). Evaporation limits work in grams: M < 5.1e14 g has evaporated today.
  - Temperatures are in GeV, and M_P = 2.435e18 GeV.
  - f_PBH is the present fraction of the dark matter.
- **Logarithms.** `lf = -inf` means no PBHs. Keep it finite-safe: `np.isfinite` checks are used throughout.
- **The peak mass stands for an extended mass function only for placement in the plane.** The limits are always applied to the full ψ. For ψ ∝ M^(−1/2) (walls from inflation, strings), the abundance sits at the lightest surviving mass.
- **Windows.** A condition named `con` is stored on disk as `_con`, because `con` is a reserved device name (`results_store._dir`).
- **Random seeds** are fixed per module. Changing the call order changes the samples slightly. `regenerate.py` therefore compares with tolerances.
- **Sampling resolution.** With N samples, p_c below 1/N is invisible. "No sample" is a statement at that resolution, not a proof.
- **Optimizers.** Two searches use differential evolution: the inflating curvaton (`infcurv_opt_de.py`) and the axion-like curvaton target search (`axion_target_de.py`). They return lower limits on maxima. The axion-like search is not reliably converged.
- **Do not put PBH limits into R.** Do not tune the distribution after looking at the PBH map.

---

## 8. Numbers to check against

Reference inflation model, ε = 1e-5 per square, statistical uncertainty ≤ 0.02.

| Quantity | (a) opt., PTA upper limit | (b) cons., PTA upper limit | (c) opt., walls make PTA signal | (d) cons., walls make PTA signal |
|---|---|---|---|---|
| (A) fraction, Gaussian | 0.20 | 0.33 | 0.22 | 0.78 |
| (A) fraction, uniform on R | 0.19 | 0.29 | 0.21 | 0.76 |
| (A) fraction, ε = 1e-6 | 0.19 | 0.29 | 0.21 | 0.76 |
| (A) fraction, ε = 1e-4 | 0.87 | 0.87 | 0.81 | 0.84 |
| (A) in every variation | 0.18 | 0.28 | 0.20 | 0.75 |
| reached in every variation | 0.11 | 0.11 | 0.15 | 0.14 |

P_reach on the reference inflation model:

| Mechanism | P_reach |
|---|---|
| FOPT, non-covariant | 0.018–0.041 |
| FOPT, covariant | 0 |
| DW, PTA as upper limit | 0.014–0.044 |
| DW producing the PTA signal | 0.022–0.39 |
| Fermi balls | 0.18 |
| walls from inflation | (1.4–1.6)e-3, all at 10^−18.6 Msun |
| strings | 0–0.024, at 10^−18.6 Msun |
| inflaton bump | (1.1–2.4)e-6 |
| axion-like curvaton | 0–3.7e-5 |
| quadratic and inflating curvaton | 0 |

Machine-readable versions:

- `pbh/measure_checks.json` (threshold curves, square sizes, squares without samples);
- `pbh/robustness.json`;
- `pbh/act_checks.json`;
- `pbh/fopt_gwvar.json`;
- `pbh/dw_bayes.json`;
- `pbh/axion_target_best.json`;
- `pbh/infcurv_de_*.json`;
- `pbh/usr_summary_fine.json`;
- `pbh/results/index.json` (P_reach and mass percentiles of every stored treatment).

---

## 9. Known limitations (state them when you build on this)

- The curvaton collapse criterion is a threshold on ζ. ζ is non-Gaussian to all orders through the nonlinear δN relation. Neither the compaction function nor peak theory is used.
- Walls from inflation are assumed not to collapse during reheating in the reference treatment. The alternative treatment is in `infwall_reh.py`.
- The limits on evaporated PBHs use analytic fits only (conservative in 3e10–1e13 g).
- The PTA analysis uses the published NANOGrav 15-year free spectrum. Pulsar-noise-model systematics are not included.
- The curvaton models and the inflaton bump are computed on the reference inflation model only.
- The mechanism set is not complete. Class (A) means "not produced by the mechanisms considered", not "impossible".

---

## 10. Citation

If you use this code or its results, cite the paper and this repository; see `README.md`.
