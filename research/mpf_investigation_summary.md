# MPF Baseline Investigation — Consolidated Summary

All DRMS in metres, horizontal, warm-up excluded. Lines: Flt1003 L1003.08 (survey),
Flt1007 L1007.06 (free-flight). Each block has its OWN EKF/FGO reference because the
signal and R differ between blocks — numbers are not comparable across blocks, only
within a block.

MPF variants:
- **MPF+TL / MPF+TL+NN (online)** = paper baseline; carries TL (and NN) in the
  conditionally-linear block, run on RAW uncompensated cabin mag (cold start).
- **plain MPF** = native map-matching RBPF (mpf_logw): no online TL, no online NN;
  fed an OFFLINE-compensated measurement. Used to isolate the estimator from the
  online joint-estimation burden.

---

## PART 1 — Paper baseline: cold start, raw uncompensated Mag 4/5 (online joint est.)

Full four-way comparison from the paper breadth table (canonical, pinned), with the
MPF+TL / MPF+TL+NN divergence magnitudes from the "fixed" recipe run (mpf_fixed) shown
in parentheses. "div." = >10 km; "err." = off map.

| Line | Mag | EKF online | EKF+TL+NN | MPF+TL | MPF+TL+NN | FGO |
|------|-----|-----------|-----------|--------|-----------|-----|
| 1007.06 | 4 | 46.7 | 48.9 | div. (3306) | 1446 | **32.7** |
| 1007.06 | 5 | 17.8 | 18.3 | 180 | 81 | **13.8** |
| 1003.08 | 4 | err. | 46.6 | div. (Inf) | 347 | **26.1** |
| 1003.08 | 5 | 21.1 | 20.7 | 152 | 4197 | **12.4** |

Two separated conclusions (paper §V):
1. **Robustness** — "causal filters diverge" is FALSE for the state of the art: the
   NN-augmented **EKF+TL+NN stays bounded on all 8 counted cases** (46–49 / 18–21 m);
   the NN is what buys causal cold-start robustness. But the **MPF+TL diverges on all
   8** — a full Bayesian representation is not enough; bounded cold-start compensation
   without a network needs the batch/window re-smoothing.
2. **Accuracy** — the FGO matches that robustness WITHOUT any NN and is more accurate
   (best of four on all 8, by 4–91 m).

Warm control — same online MPF+TL on the COMPENSATED stinger Mag 1 (mpf_warm):
1003.08 MPF+TL 2977 / MPF+TL+NN 1724 / FGO 14.6; 1007.06 43.7 / 95.6 / 16.7.
Online joint estimation on top of an already-compensated signal is ill-posed (the
extra TL states are unidentifiable) — 1003.08 still diverges.

---

## PART 2 — Is the MPF estimator core sound? (offline compensation, plain MPF)

### 2a. Clean stinger mag_1_c (offline TL-compensated), native MPF — mpf_clean
| Line | EKF | FGO | best MPF (log-w, thr0.1+rgh5, N=1000) |
|------|-----|-----|----------------------------------------|
| 1003.08 | 17.7 | 12.5 | 121.7 |
| 1007.06 | 20.0 | 16.7 | 29.4  |

Raw uncompensated Mag5 with the same native MPF: 627 / 1806 (diverges).
→ On a clean signal the RBPF navigates; on the raw cabin mag it does not.

### 2b. Offline-NN-compensated Mag 5, plain MPF at the FINAL config — mpf_np_t05
(thr 0.5 standard resampling, roughen 5, R=40 nT matched, N=4000)
| Line | comp-RMSE (nT) | EKF | MPF |
|------|----------------|-----|-----|
| 1003.08 | 49.7 | 41.2 | 36.3 |
| 1007.06 | 36.0 | 33.3 | 35.1 |

**With an accurate measurement and a proper config the plain MPF matches the EKF.**
The estimator core is sound; the cold-start failure is the online joint estimation,
not the sampler.

---

## PART 3 — MPF configuration sensitivity (why it needs care)

All on offline-NN-compensated Mag 5. Each knob isolated.

### 3a. Measurement noise R  (roughen 5, N=1000) — mpf_offnn_R
| R (nT) | 1003.08 | 1007.06 |
|--------|---------|---------|
| 12 | Inf | 118.0 |
| 25 | 331.9 | 68.0 |
| **40** | **102.3** | **40.9** |
| 80 | 964.7 | 40.2 |

R MUST match the compensation residual (~36–50 nT). R=12 (over-confident) collapses
the weights. The EKF, on the identical signal, is insensitive (38–41 / 29–33 m).

### 3b. Resample threshold  (R=40, roughen 5, N=1000) — mpf_resamp + mpf_every
| thresh | 1003.08 | 1007.06 | note |
|--------|---------|---------|------|
| 0.1 | 109.2 | 203.7 | rare resample → degeneracy |
| 0.3 | 75.1  | 435.4 | |
| **0.5** | **65.5** | **85.4** | standard ESS<N/2 — best |
| 0.7 | 250.1 | 99.7 | |
| 1.0 (every step) | Inf | 130.2 | impoverishment |

Standard ESS<N/2 is best; both extremes (rare / every-step) are worse.

### 3c. Roughening  (thr 0.5, R=40, N=4000) — mpf_np_t05 vs mpf_rough0
| roughen (m) | 1003.08 | 1007.06 |
|-------------|---------|---------|
| 0 | 626.4 | 432.8 |
| **5** | **36.3** | **35.1** |

Roughening (regularized PF) is load-bearing — without it the filter impoverishes.

### 3d. Particle count  (R=40, roughen 5) — mpf_np / mpf_np2 (thr 0.1) & mpf_np_t05 (thr 0.5)
| N | 1003.08 (thr0.1) | 1007.06 (thr0.1) | 1003.08 (thr0.5) | 1007.06 (thr0.5) |
|------|------|------|------|------|
| 1000 | 109.2 | 203.7 | 183.4 | 76.7 |
| 4000 | 68.7 | 32.9 | **36.3** | **35.1** |
| 8000 | 57.6 | — | — | — |
| 12000 | 39.5 | — | — | — |
| EKF | 41.2 | 33.3 | 41.2 | 33.3 |

MPF → EKF as N grows (finite-N Monte-Carlo variance). With standard resampling
(thr 0.5) it reaches EKF-class at N=4000; the non-standard thr 0.1 needed N=12000.

---

## Reference-implementation check — Canciani & Raquet (TAES 2017)

Our mpf_logw vs Canciani's MPF algorithm:
- Measurement predict, weight, KF update, marginalized time update: **identical**.
- **Proposal / time update = bootstrap (prior) proposal — identical** (Canciani eq 30).
  So an optimal proposal would DEPART from the reference, not "fix" it.
- Measurement-additive states: Canciani C=[…,1,1] = temporal-variation FOGM (V) +
  constant bias (c). We cover BOTH: FOGM state S (V) + TL `:bias` coefficient (c),
  factored through the online TL — cleaner given we do online compensation.
- Deviations we introduced: resample threshold (we used 0.1; Canciani every-step) and
  roughening (Canciani has none). Resolved: standard ESS<N/2 + roughening is the config.

---

## Bottom line for §V

1. **Core sound**: with an accurate measurement + proper config, plain MPF ≈ EKF
   (N=4000, 35–36 m).
2. **Finite-N penalty**: MPF < EKF is Monte-Carlo variance; MPF → EKF as N grows.
3. **Needs care EKF/FGO do not**: R matched to the residual, standard resampling,
   roughening, and ~4000 particles — four requirements, each standard/citable.
4. **Cold start**: the residual is time-varying and unknown, so R cannot be matched
   a priori → the online MPF+TL / MPF+TL+NN diverge (Part 1). The FGO's robust
   windowed MAP needs none of that care and stays bounded.
5. **Faithful to Canciani**: proposal and marginalization identical; final config =
   bootstrap + ESS<N/2 + matched R + roughening. No "crippled baseline".
