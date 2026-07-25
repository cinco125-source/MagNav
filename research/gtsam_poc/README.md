# GTSAM MagNav PoC — results summary

Reproduction of the Julia fixed-lag MagNav FGO (`src/fgo_online.jl`) in Python
GTSAM, on SGL 2020 line 1007.06 (Flt1007, Renfrew_395 map, uncompensated cabin
Mag 5, cold-start Tolles-Lawson). All scripts in this directory; environment is
a WSL venv (`gtsam h5py numpy scipy matplotlib ppigrf` — gtsam has no Windows
wheels).

## Headline table — full line, identical conditions

87.3 min (N=52401), DRMS [m] after the 10-min warm-up (`fgo_breadth.jl`
convention). Julia numbers from `research/README.md` §2b.

| Method | Type | DRMS [m] | Note |
|---|---|---:|---|
| INS only | — | 317.5 | 87-min drift |
| MPF+TL (particle filter) | causal, online | **diverged** | >10 km; 9/10 cases diverge |
| EKF (compensated stinger Mag 1) | causal, online | 20.0 | good sensor + prior compensation |
| EKF+TL+NN (paper reimpl.) | causal, online | 18.3 | NN prevents divergence, no accuracy gain |
| EKF-online (joint TL) | causal, online | 17.8 | diverges on Mag 4 |
| Julia FGO window | quasi-online (300 s window) | 13.8 | prior reference |
| **GTSAM window (this PoC)** | quasi-online (same structure) | **12.4** | reproduction, slightly better |

Same-line segment cross-check (first 10 min, warm=60 s): Julia FGO 21.4 m vs
GTSAM window 17.5 m — consistent with the full-line comparison. INS 317.5 vs
318 in the Julia table confirms export integrity.

Reading: causal filters (no past relinearization) cluster at 18–20 m and the
particle filter diverges outright, while window re-smoothing lands at 12–14 m —
the advantage is the re-smoothing, not "being Bayesian". An uncompensated cabin
sensor at 12.4 m beats the compensated tail stinger EKF (20.0 m).

## Online (per-epoch) variants — 10-min segment

`IncrementalFixedLagSmoother` (ISAM2), one `update()` per 0.1 s sample; a
causal ("realtime") estimate exists at every step. DRMS warm=60 s / 300 s:

| Variant | 60 s | 300 s | Wall clock |
|---|---:|---:|---|
| full-batch LM (offline bound) | 12.8 | 10.9 | ~3 min |
| window ×3 (Julia mirror) | 17.5 | 12.6 | 2.5 min |
| ISAM2 lag=300 s, smoothed | 14.8 | 11.0 | 13.4 h (Python callbacks) |
| ISAM2 lag=30 s, realtime, cold TL | 82.4 | 16.5 | 83 s (14 ms/epoch) |
| ISAM2 lag=30 s, realtime, warm TL | 49.5 | 15.1 | 83 s |

Cold-start transient: warm TL halves it (82→50 m) but does not remove it —
the rest is map-gradient observability (evidence: cold lag=300 s has no
transient; both runs lock in at the same ~4.5-min map feature). Practical
recipe: warm TL + lag scheduling (long early, 30 s after convergence); native
C++ factors would make lag=300 s realtime too.

## What was fixed to get here

1. **x226 indeterminate**: Qd position diag is 1e-31 rad² → whitening spans
   ~30 orders and Cholesky (squares conditioning) dies. Fix: `Qd + 1e-20·I`
   (~0.6 mm/step, negligible) + **QR factorization** for both ISAM2 and LM.
   (A 1e-8 floor + per-epoch reg priors suppress the exception but pin the
   estimate onto the INS — zero correction.)
2. **h5py axis order**: Julia column-major arrays arrive transposed — Phi
   slices (`dt/R` lands at [3,0] not [0,3]) and the square 200×200 map grid
   (silently passed a shape check; residual std 242 nT wrong vs 31 nT fixed).
3. **warm=600 s on a 600 s segment** → empty DRMS mask, `ref_drms=NaN`
   (recomputed by CI at warm=60 s → 21.4 m).

## Files

| File | Purpose |
|---|---|
| `export_line.jl` | original Julia segment export (CI) |
| `export_full_line.py` | full-line export, no Julia needed (port validated bit-exact vs the Julia segment; map grid within 0.9 nT std) |
| `run_gtsam.py` | per-epoch ISAM2 fixed-lag; CLI: `<h5> [lag_s]`; realtime+smoothed DRMS |
| `run_gtsam_window.py` | sliding-window mirror of `fgo_online_window`; CLI: `<h5> [tag]` |
| `run_gtsam_batch.py` | full-batch LM offline bound |
| `run_gtsam_warm.py` | warm-TL-prior variant: `<h5> <est_npz> [lag_s]` |
| `diag_layout.py` | verifies the axis-order fixes numerically |
| `plot_results.py` | summary figure (`gtsam_poc_results.png`) |

Raw data (not in git): `~/magnav_data/` in WSL — `Flt1007_train.h5` +
`Renfrew_395.h5` from the `Artifacts.toml` Dropbox URLs, and the generated
`line_1007_06_full.h5`.

Open follow-ups: cross-line TL transfer (true warm start), porting the
`obs_gate` observability gate (targets exactly the cold-start transient),
lag scheduling sweep.
