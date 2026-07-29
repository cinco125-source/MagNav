# TAES manuscript draft

`taes_fgo_magnav.tex` — IEEE Transactions on Aerospace and Electronic Systems
draft: *Factor-Graph Aeromagnetic Compensation for Airborne Magnetic-Anomaly
Navigation — A Neural-Network-Free Sliding-Window Alternative to Online EKF
Calibration.*

All quantitative claims come from the reproducible experiments on this branch
(`research/*.jl`, run in CI):

| Paper item | Source script | Key numbers |
|---|---|---|
| Table I (batch FGO vs EKF) | `research/fgo_benchmark.jl` | INS 114.6, EKF 28.3, FGO 14.3/14.0 m |
| Table II (cold-start line 1007.06) | `research/paper_baseline.jl`, `research/paper_impl.jl` | FGO win5min+Huber 32.6/14.2; EKF+TL+NN 40.0/17.5; paper 37/14 |
| Table III (breadth, 5 lines) | `research/fgo_breadth.jl` | FGO wins 8/9; EKF diverges on Mag 4 |
| Sensor-error factors (sim) | `research/fgo_sensor_ablation.jl` | 36.6 → 4.8 m (−87%) |
| Observability discussion | `research/observability*.jl`, `research/OBSERVABILITY.md` | sweet spot; exogeneity rule (honest negatives) |

## Files

- `taes_fgo_magnav.tex` — LaTeX source (IEEEtran, `\cite` + `references.bib`),
  written to the Paper-Orchestra style rules (Abstract ≤2 numbers, ≤4 subsections
  per section, no self-praise, a Proposition + proof for the observability
  condition).
- `references.bib` — bibliography. DOIs are included only where confidently known;
  the rest are left for manual verification (see citation check below).
- `make_figures.py`, `plot_tracks.py` — figure generation (see below).
- `figs/` — the six vector-PDF figures included by the manuscript.
- `taes_fgo_magnav.pdf` — the compiled draft (committed by the `Paper PDF` CI job).

## Figures

Six vector PDFs in `figs/`, regenerated from the CI numbers and committed CSVs:

| Figure | Script | Content |
|---|---|---|
| `fig_graph.pdf` | `make_figures.py` | Factor-graph schematic (shared TL variable) |
| `fig_window.pdf` | `make_figures.py` | Fixed-lag sliding window (commit stride + look-ahead) |
| `fig_coldstart.pdf` | `make_figures.py` | Line 1007.06 FGO vs EKF+TL+NN (Table I) |
| `fig_breadth.pdf` | `make_figures.py` | 5-line breadth, log DRMS, divergence (Table II) |
| `fig_map.pdf` | `plot_tracks.py` | Eastern anomaly map + flight line 1003.02 + zoom inset |
| `fig_poserr.pdf` | `plot_tracks.py` | Position-error curves (INS / EKF / FGO) |

```
python make_figures.py    # schematic + bar figures
python plot_tracks.py      # geographic figures from track_data.csv / map_grid.csv
```

## Building the PDF

**In CI (authoritative):** the `Paper PDF` workflow
(`.github/workflows/paper_pdf.yml`) regenerates the figures, compiles with the
real `IEEEtran.cls` from TeX Live (`xu-cheng/latex-action`), and commits
`taes_fgo_magnav.pdf` back to the branch. This is how the genuine
IEEEtran-formatted PDF is produced (local `pdflatex` is unavailable in the dev
sandbox).

**Locally (requires a TeX distribution with `IEEEtran.cls`):**

```
python make_figures.py && python plot_tracks.py
pdflatex taes_fgo_magnav && bibtex taes_fgo_magnav && \
pdflatex taes_fgo_magnav && pdflatex taes_fgo_magnav
```

## Citation integrity (Wave 4, before submission)

```
python check_citations.py . --no-net     # offline: \cite↔bib matching only
python check_citations.py .               # + DOI existence via doi.org
```

Current status: `missing 0, unused 0` (no fatal issues); 8 entries are NO-DOI
(manual verification needed — DOIs were not fabricated).

## Status / TODO before submission

- Authors, affiliations, acknowledgments.
- Figures: geographic tracks + position-error plots (`research/fgo_tracks.jl`
  outputs `fgo_map_track.png`, `fgo_pos_error.png`) can be dropped in as
  `\includegraphics`.
- Verify/complete the Hager et al. (2026) citation metadata against the published
  version.
- Optional: Monte-Carlo statistics, MPF / grid-MMSE baselines, real sensor-error
  validation (see paper §VII limitations).
