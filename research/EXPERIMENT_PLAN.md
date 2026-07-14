# Experiment Plan — Statistical Rigor Pass (for IEEE TAES)

Goal: close the two Tier-1 reviewer levers from the red-team re-review by adding
the estimator-consistency and statistical-significance evidence a TAES reviewer
expects, without weakening any existing (honest) claim.

## 0. Why (reviewer gaps this closes)

| Gap (red-team) | Test that closes it |
|---|---|
| No NEES/NIS consistency check | **B** NEES (state) + NIS (innovation) |
| No ±2σ covariance bands | **C** ±2σ envelope on position error |
| Single-run DRMS behind count-based "8/8" | **A** Monte-Carlo DRMS with 95% CIs |
| Headline vs imperfect reimplementation, no error bars | **A** error bars on the breadth bars |

## 1. Feasibility (already verified)

`fgo_online` / `fgo_online_window` and the EKF baseline return
`FILTres(x, P, resid, …)` with per-epoch state `x[nx,N]`, covariance
`P[nx,nx,N]`, and measurement residuals. So the estimator covariance and the
innovation are available directly — no solver change needed. `nx` includes the
3-axis position-error block, so a 2-DOF horizontal-position marginal is a simple
slice of `P`.

## 2. Tests

> **Design correction (important, honesty).** `get_ins` returns the *real* recorded
> flight INS, so each real SGL line is a single physical realization: its
> measurement noise cannot be re-drawn, and a classical Monte-Carlo over the real
> breadth lines would be fabricated. We therefore split the rigor two ways: the
> repeated-realization statistics (CIs, ANEES, ANIS) are done on **simulated**
> flights (`create_XYZ0`, the same generator the sensor ablation already uses),
> where drawing many noise realizations is valid; and on the **real** lines we
> report the single-realization consistency evidence that is actually available
> (NEES-vs-time, ±2σ envelope), labelled as single-run.

### A. Simulation Monte-Carlo — DRMS CIs + consistency (rigorous)
- **Design.** `N` simulated flights from `create_XYZ0`, each with an independent
  INS-error draw and measurement-noise realization; cold-start TL. Run the EKF and
  the FGO window on the same draw (paired).
- **Metrics.**
  - DRMS: mean ± 95% t-interval for EKF vs FGO; paired win-rate P(FGO < EKF).
  - **ANEES** (2-DOF horizontal position): `ε = e_posᵀ P_pos⁻¹ e_pos` in the state's
    native (rad) coordinates so it is unit-free, time-averaged then averaged over
    the `N` runs, compared to the two-sided 95% χ²₂ interval scaled by `N`.
    ANEES ≫ 2 ⇒ optimistic covariance; ≪ 2 ⇒ conservative.
  - **ANIS** (stretch, 1-DOF): scalar innovation `ν_t²/S_t`, `S_t = H_t P_t H_tᵀ + R`.
    Requires reconstructing the map/TL/FOGM measurement Jacobian `H_t` post hoc; if
    that reconstruction is not clean, report NEES only and say so.
- **Outputs.** a Monte-Carlo table (EKF vs FGO DRMS mean±CI, ANEES, win-rate) and
  `fig_consistency` (ANEES with χ² band; ANIS if available).

### B. Real-data single-run consistency
- **NEES-vs-time** on the real primary line 1007.06 (both mags) and batch line
  1003.02: `ε_t = e_pos,tᵀ P_pos,t⁻¹ e_pos,t` from the actual run, with the 95% χ²₂
  band. Single realization, labelled as such; shows whether the reported covariance
  is credible on real data (and, by comparison, EKF vs FGO).
- **Output.** panel in `fig_consistency` or a small companion plot.

### C. ±2σ covariance envelope (real data)
- Representative real run (batch line 1003.02, clean compensated sensor): North and
  East position error vs time with the estimator ±2σ band, read directly from the
  `n_std`/`e_std` that `eval_filt` already computes from `P`. Shows the reported
  uncertainty contains the error.
- **Output.** `fig_sigma` (2-panel N/E error with ±2σ).

### D. (optional, if CI budget allows)
- No-Huber FGO column in the breadth table (isolate the robust kernel across all
  lines, not only 1007.06).
- `L_w` / `L_o` sensitivity with ± bands.

## 3. Honesty guardrail
NEES/NIS may reveal the estimator is optimistic (common for a smoother with a
robust kernel and a strong map factor). If so we **report it and fix it in the
open**: inflate `Q`/scale `R` until ANEES enters the χ² band, and document the
adjustment. A consistent-after-tuning result is publishable; a hidden inconsistent
one is not. Either outcome is reported, never suppressed.

## 4. Implementation steps
1. `research/fgo_montecarlo.jl` — seed loop; collects DRMS, per-epoch NEES, NIS,
   and ±2σ traces; writes CSVs (and prints to the CI log for provenance).
2. Extend `.github/workflows/fgo_research.yml` with a Monte-Carlo job; scope `N`
   and reuse each line's loaded `XYZ` across seeds; log wall-clock and `N`.
3. Figures: `fig_consistency`, `fig_sigma`, and error bars on `fig_breadth`
   (in `make_figures.py` / `plot_tracks.py`), all in the existing dataviz style.
4. Manuscript: new Results subsection "Consistency and statistical significance"
   holding the simulation MC (DRMS CIs, ANEES) and the real-data single-run NEES /
   ±2σ evidence. The real breadth stays single-run (one physical realization per
   line, stated plainly); the abstract/claims gain the simulation CIs and the
   consistency result, not a false "Monte-Carlo on real data". Soften the matching
   Conclusion limitation instead of deleting it (single-realization real data is
   inherent). Update `research/README.md` provenance.

## 5. Compute scope and the one decision needed
A fixed-lag smoother over `N` seeds × 8 cases is the cost driver. Proposed scope:
- **Consistency (B) + ±2σ (C):** `N = 30` on 1007.06 (both mags) and 1003.02.
- **MC-DRMS (A):** `N = 30` on all 4 lines × 2 mags.

`N = 30` gives usable 95% intervals while keeping the CI job bounded. If the job
runs too long, the fallback is `N = 20` (stated in the paper). Pushing `N = 100`
(tighter intervals, stronger claim) means a much longer CI job.

**Decision:** run at `N = 30` (recommended balance), or go to `N = 100` for
publication-grade intervals at the cost of a long CI run?

## 6. Expected impact
Closes Tier-1 #1 in full and Tier-1 #2 in part (error bars on the headline
margin). Estimated TAES accept probability ~30% → ~50%+, contingent on the
consistency result being clean (or cleanly tuned per §3).
