# Where primordial black holes can come from — code and analysis notes

Code and documentation for

> T. Matsuda, *Where primordial black holes can come from: how likely formation mechanisms are to produce a given mass and abundance* (2026).

The analysis treats the black-hole mass M and the dark-matter fraction f_PBH as outputs of formation mechanisms, rather than free parameters. For ten mechanisms it computes the probability that each square of the (M, f_PBH) plane is populated. The parameter distributions are fixed by non-PBH observations: the CMB, the gravitational-wave backgrounds bounded or measured by LIGO–Virgo–KAGRA and NANOGrav, and BBN.

## Contents

| Path | What it is |
|---|---|
| `docs/analysis_notes.pdf` (`.tex`) | **Analysis notes.** For every mechanism: formulas, parameter ranges and conditions. Also the external data, statistical procedures, construction of every figure, analytic estimates checked against the numerical results, and the commands that reproduce them. |
| `AI_GUIDE.md` | **Guide for AI assistants.** Conventions, data format, the interface a formation mechanism must provide, recipes, and reference numbers. Load it into an AI assistant to follow the analysis or to repeat it for other models. |
| `pbh/` | All code (Python 3.10+, numpy ≥ 2.0, scipy, matplotlib). Table I of the analysis notes maps files to parts of the analysis. |
| `expected/` | Reference values used by `pbh/regenerate.py` to check a rebuild. |
| `external/` | Third-party data (see below). |

## Quick start

```bash
bash setup.sh            # or: powershell -ExecutionPolicy Bypass -File setup.ps1
cd pbh
python regenerate.py     # rebuilds the main results (1.5-2.5 h on 8 cores) and compares with expected/
```

The additional steps behind the variations and checks of the paper are listed in Sec. X of the analysis notes.

## Stored samples

The stored samples of all mechanisms, treatments and variations (about 6 GB) are deposited on Zenodo: doi:10.5281/zenodo.XXXXXXX (to be inserted). With them the figures can be redrawn without rerunning the mechanisms:

```bash
cd pbh
python get_data.py ~/Downloads/pbh-data-results.zip ~/Downloads/pbh-data-bgscan.zip ~/Downloads/pbh-data-samples.zip
python fig_combined.py && python measure_checks.py && python measure_plots.py
```

The repository itself contains the code, the small inputs that `regenerate.py` needs, the summary files of every stored result (`pbh/results/*/*/meta.json`, `pbh/results/index.json`) and the result summaries listed in `AI_GUIDE.md`, Sec. 8.

## Third-party data

| Data | Source | Version |
|---|---|---|
| PBH limits | PBHbounds, B. J. Kavanagh, https://github.com/bradkav/PBHbounds (MIT licence) | commit 96bc091 |
| PTA free spectra (EPTA DR2, used for comparison) | ceffyl, https://github.com/astrolamb/ceffyl (MIT licence) | commit 0cd2b3a |
| NANOGrav 15-year KDE free spectrum | Zenodo, doi:10.5281/zenodo.10344086 | v1.0.0 |

Cite the original publications of these data sets when you use them; the analysis notes give the references.

## Citation

Please cite the paper (reference to be added on publication) and this repository.

## Licence

Code (`pbh/`, setup scripts): MIT licence, see `LICENSE`. Documents (`docs/`, `README.md`, `AI_GUIDE.md`): Creative Commons Attribution 4.0 International (CC BY 4.0), see `docs/LICENSE`. Third-party data in `external/` keep their own licences (`external/README.md`). The stored samples on Zenodo are released under CC BY 4.0.
