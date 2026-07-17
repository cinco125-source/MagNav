# Factor Graph Optimization for Airborne Magnetic Navigation

Research extension to **MagNav.jl** that reformulates airborne magnetic anomaly
navigation as **factor graph optimization (FGO)** — batch maximum a posteriori
(MAP) estimation — and extends it with online Tolles-Lawson compensation and
physically-modeled scalar-magnetometer sensor-error factors.

Branch: `claude/fgo-problem-research-bllgbr` · all results below are produced by
the CI workflow `.github/workflows/fgo_research.yml` on real SGL 2020 flight data
and reproducible simulations.

---

## 1. What was built

| File | Lines | What it is |
|---|---:|---|
| `src/fgo.jl` | 514 | `fgo` — batch MAP smoother. Two equivalent solvers: iterated **RTS** (`solver=:rts`) and global **sparse Gauss–Newton / QR** (`solver=:gn`, square-root SAM). Robust Huber/Cauchy IRLS kernels. Reuses the existing Pinson model (`get_Phi`/`get_H`/`get_h`). |
| `src/fgo_online.jl` | 257 | `fgo_online` — batch FGO with **Tolles-Lawson coefficients as factor-graph variables** (joint aeromagnetic compensation + navigation; the batch analog of `ekf_online`). |
| `src/fgo_sensor.jl` | 272 | `fgo_sensor` — appends **physically-modeled sensor-error states**: OPM heading error (cosine harmonics {1,2,4} in the sensor–field angle ψ; Hager et al. 2026, Wang et al. 2020 GRSL), fluxgate hard-iron bias (`m·û_body`), linear drift, and OPM equatorial dead-zone weighting (`R·max(|cos ψ|,ε)²`). |
| `src/eval_filt.jl` | +34 | `run_filt` dispatch for `:fgo` and `:fgo_online` with `solver`/`robust`/`n_iter` options. |
| `src/MagNav.jl` | +5 | include + export `fgo`, `fgo_online`, `fgo_sensor`; add `SparseArrays` dep. |
| `test/test_fgo.jl` | 254 | structure, accuracy-vs-EKF/INS, GN↔RTS agreement, robust-kernel, `fgo_online`, and `fgo_sensor` tests (registered in `runtests.jl`). |
| `examples/fgo_example.jl` | 65 | runnable EKF-vs-FGO demo on the bundled simple dataset. |
| `research/fgo_benchmark.jl` | 181 | real SGL **Flt1003** DRMS benchmark across all methods. |
| `research/paper_baseline.jl` | — | line 1007.06: our FGO-online (static + sliding-window TL) vs Hager et al. (2026) cited DRMS. |
| `research/paper_impl.jl` | — | **re-runs** the paper's online EKF+TL+NN (`ekf_online_nn`) cold start on line 1007.06 — a genuine reproduced baseline (Mag 4 40.0 m, Mag 5 17.5 m). |
| `research/fgo_breadth.jl` | — | **breadth**: window FGO vs causal EKF-online on 5 lines / 3 flights / 2 maps, cold-start cabin mags (§2b). The paper reports the 4 navigation/survey lines (8 cases, FGO best of 4 on 8/8) and sets aside the 14-min calibration line 1006.08. Baselines: weak (online-TL EKF) + strong (EKF+TL+NN) + fair particle filter (MPF+TL); NN-free. |
| `research/fgo_montecarlo.jl` | — | **consistency & significance**: 30-seed simulation Monte-Carlo (fixed sim trajectory, per-seed INS-error + clean-measurement re-draw) of EKF vs the marginalized particle filter (MPF) vs batch FGO on the clean/compensated sensor. Reports DRMS mean±95% CI (FGO 1.4±0.2, EKF 4.1±0.4, MPF 4.9±0.8 m) and per-axis 2-DOF position ANEES (FGO 1.57, EKF 1.72 — both conservative; MPF 59.8 — its particle covariance is badly over-confident under particle depletion). Real lines are single-realization so MC is simulation-only. Outputs `montecarlo_{summary,nees,sigma}.csv`. |
| `research/fgo_sensor_ablation.jl` | 154 | factorial sensor-error ablation with injected-truth recovery. |
| `research/fgo_tracks.jl` | 131 | geographic map+track and position-error figures. |
| `.github/workflows/fgo_research.yml` | — | CI: test suite + all three research scripts on every push. |
| `docs/src/nav.md` | +41 | API documentation sections. |

**Theory in one line.** For the linear-Gaussian Pinson error model the factor
graph is a chain, so the MAP estimate equals the fixed-interval (RTS) smoother;
the novelty is not the smoothing but **modeling sensor-error physics as graph
variables estimated jointly with navigation** — which a position-only grid /
point-mass estimator structurally cannot do.

---

## 2. Results (real SGL Flt1003, line 1003.02, Eastern_395 map, 63 min)

Horizontal position **DRMS** [m], lower is better:

| Method | DRMS | Runtime |
|---|---:|---:|
| INS (no aiding) | 114.6 | — |
| EKF (baseline) | 28.3 | 23 s |
| **FGO (RTS)** | **14.3** | 18 s |
| **FGO (RTS + Huber)** | **14.0** | 16 s |

**Batch solver check (10-min segment):** EKF 47.9 m; FGO-RTS 18.4 m vs global
sparse GN/QR 19.4 m (18.4 m with Huber) — the two solvers converge to the same MAP
estimate (RTS is ~2× faster on
the chain).

**Online Tolles-Lawson (uncompensated cabin Mag 4):**

| Method | DRMS |
|---|---:|
| EKF-online | **diverged (2.8×10⁵ m)** |
| FGO-online | 60.1 |
| **FGO-online + Huber** | **22.6** |

→ Under an uncompensated magnetometer the causal EKF-online diverges while batch
FGO-online with a robust kernel holds 22.6 m — better than the compensated-stinger EKF.

---

## 2b. Breadth: the FGO advantage is not a one-line fluke

The headline comparison (§5b) is on one line. To test generality we run the **same
cold-start Tolles-Lawson model** through the causal filter (`ekf_online`) and the
factor graph (`fgo_online`, fixed-lag 5-min window + Huber) on **5 long survey
lines across 3 flights and 2 maps**, on the uncompensated cabin magnetometers
Mag 4 / Mag 5. No neural network anywhere — the only difference is causal EKF vs
batch/window smoothing, so a consistent FGO advantage isolates the factor-graph
formulation itself. DRMS [m] after a 10-min warm-up (`research/fgo_breadth.jl`):

Three baselines — weak (online-TL EKF), strong (reimplemented EKF+TL+NN, same
recipe as `paper_impl.jl`), and a fair particle filter (**MPF+TL**: the
marginalized/Rao-Blackwellized PF already in MagNav.jl, extended to carry the
Tolles-Lawson coefficients in its conditionally-linear-Gaussian block,
`research/mpf_online.jl`) — vs the NN-free FGO window. DRMS [m] after a 10-min
warm-up (`research/fgo_breadth.jl`); ✗ = diverged beyond the 10 km cutoff:

| flight | line | map | mag | INS | EKF-online | EKF+TL+NN | MPF+TL | **FGO window** |
|---|---|---|---|---:|---:|---:|---:|---:|
| Flt1003 | 1003.02 | Eastern | Mag 4 | 124 | **41626 ✗** | 99.1 | **✗** | **42.6** |
| Flt1003 | 1003.02 | Eastern | Mag 5 | 124 | 28.1 | 29.5 | **✗** | **21.7** |
| Flt1003 | 1003.08 | Renfrew | Mag 4 | 272 | **off-map ✗** | 46.6 | **✗** | **26.1** |
| Flt1003 | 1003.08 | Renfrew | Mag 5 | 272 | 21.1 | 20.7 | **✗** | **12.4** |
| Flt1006 | 1006.08 | Eastern | Mag 4 | 198 | **17179 ✗** | 952 | **✗** | **193.9** |
| Flt1006 | 1006.08 | Eastern | Mag 5 | 198 | 117.5 | **108.3** | 793.7 | 122.0 |
| Flt1007 | 1007.02 | Eastern | Mag 4 | 121 | **35482 ✗** | 130.0 | **✗** | **38.6** |
| Flt1007 | 1007.02 | Eastern | Mag 5 | 121 | 31.6 | 30.4 | **✗** | **14.5** |
| Flt1007 | 1007.06 | Renfrew | Mag 4 | 318 | 46.7 | 48.9 | **✗** | **32.7** |
| Flt1007 | 1007.06 | Renfrew | Mag 5 | 318 | 17.8 | 18.3 | **✗** | **13.8** |

**EKF+TL+NN diverged on 0 cases (the NN keeps the causal filter bounded), yet the
NN-free FGO window is best of the four on all 8 counted cases** (the four
navigation/survey lines 1007.06, 1007.02, 1003.02, 1003.08). The only case where it
is not best, 1006.08 Mag 5, is on the 14-min calibration line 1006.08 (Flt1006),
which the paper sets aside and does not tabulate. The plain EKF-online **diverges to tens of km** (41626 /
35482 / 17179 m) on three Mag-4 lines and runs off-map on a fourth, while both the
NN filter and the FGO window stay bounded — the fixed-lag smoother re-linearizes
over each window, so early
navigation is protected by calibration that only becomes observable later, which a
one-pass causal filter cannot do. FGO even beats the compensated-stinger EKF on
several lines (e.g. 1003.08 Mag 5 12.4 vs 17.7; 1007.06 Mag 5 13.8 vs 20.0).

**The particle filter does not rescue the cold start.** MPF+TL — a genuine
Bayesian estimator, not a linearized filter — **diverges past 10 km on 9 of 10
cases**, and on the sole line where it stays finite (1006.08 Mag 5) it is 794 m,
far worse than every other method. Its per-particle map-matching cannot recover
from a cold-start TL that only becomes observable later: the marginalized block
updates the coefficients recursively but never re-linearizes past epochs, so the
early-flight interference is baked into the trajectory just as it is for the
causal EKF. This isolates the mechanism — the FGO advantage is the batch/window
**re-smoothing**, not merely "using a Bayesian filter." Honest caveat: on the
short 14-min line (Flt1006 1006.08) every method is weak and FGO wins only on
Mag 4 — short lines carry little map information for any estimator.

---

## 3. Sensor-error factor ablation (simulated, Eastern_395, injected truth)

Cumulative factors; INS reference 31.1 m. OPM error model (Hager et al. 2026, Wang
et al. 2020 GRSL): heading c1=10/c2=6/c4=3 nT cosine harmonics {1,2,4},
hard-iron m=[15,−10,6] nT, drift 0.02 nT/s, dead-zone noise ∝ 1/max(|cos ψ|,0.1).

| Model | DRMS | notes |
|---|---:|---|
| baseline (no sensor states) | 42.9 | worse than INS — unmodeled error corrupts aiding |
| + heading {1,2,4} | 18.9 | largest single drop |
| + dead-zone weighting | 15.5 | load-bearing (heteroscedastic OPM dead-zone noise) |
| + fluxgate bias | 8.1 | |
| **+ drift (full model)** | **6.2** | −86% vs baseline; below INS |

Huber column: 42.8 / 18.3 / 13.6 / 7.8 / 7.3 (agrees within ~2 m).

**Parameter recovery (full model vs truth).** Drift 0.018 vs 0.020 ✓ recovers
cleanly; heading c1/c2 (3.8/0.6 vs 10/6) are **observability-limited** because the
simulated flight sweeps the sensor–field angle ψ by only 12° (level flight at
~70° inclination). Honest finding: the **navigation gain is robust, but clean
sensor calibration recovery needs wider attitude excitation** — the gain comes
from the fitted combination over the observed angle range, not the individual
coefficients. Model-matched injection (simulation only).

---

## 4. How to run

```julia
using MagNav
# ... build traj, ins, meas, itp_mapS, (P0,Qd,R) as in examples/fgo_example.jl
filt = fgo(ins, meas, itp_mapS; P0, Qd, R, n_iter=5)          # batch MAP smoother
filt = fgo(ins, meas, itp_mapS; solver=:gn)                    # sparse GN/QR solver
filt = fgo(ins, meas, itp_mapS; robust=:huber)                # robust kernel
res  = fgo_online(ins, meas, flux, itp_mapS, x0_TL, P0, Qd, R) # + TL compensation
res  = fgo_sensor(ins, meas, itp_mapS; n_harm=2, cal_bias=true,# + sensor-error
                  drift=true, dead_zone=true)                  #   factors
# or via the standard pipeline:
run_filt(traj, ins, meas, itp_mapS, :fgo; P0, Qd, R)
```

Reproduce the studies:
```
julia --project=. research/fgo_benchmark.jl        # real Flt1003 DRMS table
julia --project=. research/fgo_sensor_ablation.jl  # factorial sensor ablation
julia --project=. research/fgo_tracks.jl           # map + track + error figures
```

---

## 5. Four research thrusts (status)

1. **Batch MAP FGO vs EKF on real data** — ✅ Flt1003: 28.3 → 14.0 m.
2. **"Textbook" sparse GN/QR solver** — ✅ square-root SAM form, matches RTS.
3. **Tolles-Lawson factors (joint compensation)** — ✅ EKF-online diverges, FGO-online 22.6 m.
4. **Physical OPM sensor-error factors (heading/dead-zone/bias/drift) + robust** — ✅ cumulative ablation 42.9 → 6.2 m, with an honest observability caveat.

## 5b. Comparison to the online EKF+TL+NN cold-start literature

We compare against the online EKF + Tolles-Lawson + neural-network cold-start
calibration of Hager et al. (2026, arXiv 2603.08265), on that paper's primary
line 1007.06, full 87 min, uncompensated cabin magnetometers, DRMS after a
10-min warm-up. Two scripts:

- `research/paper_baseline.jl` runs **our** methods (FGO-online with static and
  sliding-window / adaptive TL) and prints the paper's published DRMS as a cited
  reference line.
- `research/paper_impl.jl` actually **re-runs the paper's filter family**: the
  MagNav.jl `ekf_online_nn` — an online EKF with the TL basis as NN input
  features and the NN weights carried as EKF states, learned online from a cold
  start (the reference implementation of the SGL/AFIT online NN-in-EKF lineage
  the paper builds on).

**Reproducing the cold start took care, and the failure modes were instructive.**
A naive cold start of `ekf_online_nn` diverges; getting it into the paper's band
required three fixes, each addressing a distinct, diagnosable failure:

1. **Bias handling** — the NN compensation must represent only the aircraft
   interference (the core + map field is already supplied by `get_h`), and its DC
   offset is initialized from onboard data (median of `mag_uc − get_h` over the
   first minutes). Without this the first residual is the full interference DC and
   the Kalman gain spikes to a NaN divergence.
2. **Feature design** — the scalar `mag_uc` must be **excluded** from the NN
   inputs (TL A-matrix only). It carries the map anomaly, so feeding it to the
   compensation NN lets the network subtract the very signal we navigate on,
   collapsing observability (residual → 0 while position drifts to km scale — Mag 4
   at 5.8 km with `max|resid|` only 78 nT). This is exactly the instability the
   paper's natural-gradient stabilization is designed to prevent; here it is
   prevented structurally, through the feature set.
3. **Covariance** — de-trust the map through the cold-start transient
   (`meas_var = 12²`) and set the NN weight process noise by hand.

**Reproduced result (our re-run of the online EKF+TL+NN, cold start, line 1007.06,
full length; INS 318 m, EKF on compensated Mag 1 ≈ 20 m for reference):**

| Magnetometer | paper TL-only | paper TL+NN | **EKF+TL+NN re-run (ours)** | **FGO-online win 5min +Huber (ours)** |
|---|---:|---:|---:|---:|
| Mag 4 (uncompensated) | 58 m | 37 m | **40.0 m** | **32.6 m** |
| Mag 5 (uncompensated) | 15 m | 14 m | **17.5 m** | **14.2 m** |

Reading it honestly: our re-run of the online EKF+TL+NN lands **in the paper's
published cold-start band** — beating their TL-only and within a few metres of
their tuned TL+NN. It is a fair reproduction, not an exact one: we do not have
their released architecture / natural-gradient stabilization, so we sit a few
metres above their tuned filter. It now serves as a genuine, re-run baseline
rather than a cited number. Notably, our **sliding-window FGO-online** (adaptive
TL, fixed-lag smoother, *no neural network*) matches or beats that reproduced
EKF+TL+NN on the same line — see §5c. Exact numbers are in the CI artifacts
`paper_impl_results.csv` and `paper_baseline_results.csv`.

## 5c. Sliding-window (fixed-lag) FGO-online vs static-batch TL

A single batch solves one static TL coefficient set for the whole flight, which
underfits time-varying interference over a long line. Running `fgo_online` as an
**iSAM2-style fixed-lag smoother** (kwargs `win`/`overlap`; each window commits
its leading `stride` and carries the full state + covariance forward as the prior
for the next) makes the TL compensation **adapt** along the flight. On line
1007.06 (full 87 min) this recovers the long-line performance dramatically:

| Method (line 1007.06, full length) | Mag 4 | Mag 5 |
|---|---:|---:|
| FGO-online **batch** (static TL) | 123.7 m | 68.1 m |
| FGO-online **win 5 min** (adaptive TL) | 37.0 m | 15.1 m |
| FGO-online **win 2 min** (adaptive TL) | 45.9 m | 17.1 m |
| **FGO-online win 5 min + Huber** | **32.6 m** | **14.2 m** |
| — paper TL+NN (Hager et al. 2026) | 37 m | 14 m |

The fixed-lag window takes the static batch from 124 → 32.6 m (Mag 4) and
68 → 14.2 m (Mag 5), **matching or beating the paper's cold-start TL+NN with no
neural network** — the adaptive-TL relinearization plays the role their online NN
plays. Produced by `research/paper_baseline.jl`.

## 6. Reproducibility

CI runs the full test suite (Julia LTS + latest) plus all research scripts on
every push and uploads the result CSVs as artifacts. Figures are generated from
the same runs. Nothing here depends on private data — SGL 2020 and the Ottawa
maps download automatically via lazy artifacts.

## 6b. Supporting analysis: observability of joint compensation + navigation

`research/OBSERVABILITY.md` records an exploratory study of *when* joint
cold-start compensation + navigation is well-posed (the collapse seen when an
endogenous, map-carrying feature enters the compensation basis). It is honest
about its negative results: several candidate scalar predictors — the map-value
confound ρ², its out-of-sample (cross-segment) form, and an FGO-native
map-gradient confound ρ_obs — each fail to cleanly separate the safe from the
collapse-prone bases, and a ρ_obs-gated `fgo_online` (`obs_gate=true`) did not
recover an injected collapse in the linear regime. This is **supporting analysis
and future work, not a headline result**; the paper's contribution is the FGO
navigation performance above. The two robust, useful findings that survive are
qualitative: (i) cold-start joint estimation has an expressiveness *sweet spot*
bounded by under-compensation and observability collapse, and (ii) compensation
should be driven by **exogenous** (attitude/fluxgate, position-independent)
features — which is exactly what every FGO result here uses.

## 7. Limitations & next steps

- **Breadth** ✅ addressed in §2b (4 counted navigation/survey lines + 1 set-aside
  calibration line, 3 flights, 2 maps); still worth extending to Monte-Carlo
  statistics and the full SGL line set.
- **Baselines** ✅ MPF added: a marginalized/Rao-Blackwellized particle filter,
  extended to carry the Tolles-Lawson coefficients (MPF+TL), now runs as a fair
  Bayesian baseline in both studies — DRMS-competitive with the EKF but
  over-confident on the clean sim (ANEES ≈ 60, §MC) and divergent on 9/10
  cold-start cabin-mag lines (§2b). Still open: a grid/point-mass MMSE estimator
  for a like-for-like map-matching comparison.
- **Short lines**: the 14-min line (Flt1006 1006.08) is hard for every method —
  characterize the map-information floor vs line length.
- **Observability**: turn the qualitative sweet-spot / exogeneity findings
  (§6b) into a quantitative identifiability (CRLB/PCRB) condition — open.
- **Sensor calibration**: rerun `fgo_sensor` with a wide-θ maneuver profile to
  convert the weak parameter recovery into clean recovery.

## 8. Follow-up experiments (2026-07-17) and the paper-scope decision

Three follow-ups probed how far the graph extensions carry on real data. All
outputs are committed CSVs / CI artifacts.

**8a. Current-channel mechanism check** (`fgo_current_corr.jl`, line 1003.02):
the 14 recorded current channels linearly explain **28–39 %** of the raw cabin
interference (strobe, fuel pump, INS heater lead) but only **2–6 %** of the
post-fit windowed-FGO residual — the time-varying β + FOGM S absorb the
current-aligned component almost entirely — and Huber down-weighting concentrates
**2.6–3.6×** on current-switching epochs (strobe activity ↔ |residual| corr 0.55
on Mag 5). Complete result; cited in the paper (§VI-C).

**8b. Sensor-error factors on real data** (`fgo_sensor_real.jl`, compensated
Mag 1, cumulative ablation): **mixed and line-dependent** — monotone improvement
on 1003.02 (14.7 → 12.7 m, −14 %) but monotone degradation on 1007.06
(17.4 → 21.6 m, +24 %). Interpretation: the OPM error terms are defined in the
sensor–field angle ψ, which depends on an **assumed optical-axis orientation**
(body-x here); SGL documents neither the axis nor the sensor's error datasheet,
so blind application turns dead-zone weighting into an arbitrary attitude-dependent
reweighting — helps by luck on one line geometry, hurts on another. Lesson:
sensor-error factors need documented sensor metadata; not claimable on SGL.

**8c. Current-augmented compensation basis** (`fgo_current_basis.jl`,
`A_extra` = causally z-scored current channels, 4 counted lines × Mag 4/5):
random-walk γ **fails 8/8** (validates the physics: coupling geometry is fixed →
γ must be static); static γ and attitude-modulated triplets improve **6/8**
(−5 to −19 %, e.g. 1007.06 Mag 4 32.7 → 26.6 m) but **fail 2/8 badly**
(Eastern Mag 4 cases, +65–88 %). Root cause of the failures: causal first-10-min
z-scoring explodes the scale of channels that are quiet during warm-up
(γ̂ = −10 964 nT/A for the fuel pump on 1003.08 is the smoking gun); γ̂ cross-line
replication is partial (com radio −186/−129/−85 nT/A: sign and order reproduce;
switching channels do not). Promising but not robust — needs principled channel
scaling/gating before it is claimable.

**Paper-scope decision**: the manuscript now makes a single claim — cold-start
aeromagnetic compensation as **joint estimation in a windowed factor graph** —
with 8a kept as the real-data mechanism check. The simulated sensor-error factor
study (§3 above) and 8b/8c are recorded here and in the code but removed from the
paper's contribution list; the conclusion states the honest condition (documented
sensor metadata; principled telemetry scaling) under which each extension becomes
claimable. `src/fgo_sensor.jl` and all experiments remain in the repo.

See the commit history on this branch for the full development trail.
