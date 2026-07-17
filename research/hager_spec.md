# Hager et al. 2026 (arXiv 2603.08265v1) — faithful-reimplementation spec

Extracted from the full preprint (87 pp) for a to-the-letter reimplementation
(`hager_impl.jl`, planned). Our existing reproduction (`ekf_tlnn.jl` /
`paper_impl.jl`) is a *recipe-level* reproduction; the items below are the exact
published choices, with deltas from our version flagged.

## Filter

- **Error-state EKF**, state dimension **(40 + Np)**:
  `[δp(3); δv(3); δψ(3); baro loop (from Gnadt's thesis); accel bias(3);
  gyro bias(3); S_CB(1, constant measurement bias); β_TL(18); Λ_NN(Np);
  m(3, vector-magnetometer states)]`
- Vector magnetometer treated as **pseudo-control input**: states `m` driven by
  the measurement through a first-order lag with τ→0 (discrete input matrix → I).
- F = diag(F_NAV, F_IMU, F_MAG, F_TL, F_NN, F_VMAG); TL and NN blocks are
  constants (random-walk via Q only).
- **Innovation gating**: reject measurement if NIS > χ² threshold **6**,
  active **after the first 10 min** (no gating during the transient!).

## Calibration model

- Hybrid **additive** TL + NN: interference = A_t'β_TL + NN(ϕ_t) · α, with
  **α = 400 nT** output denormalization (no offset).
- TL: 18 terms (permanent+induced+eddy, no bias column — S_CB is a separate state).
- NN: **single hidden layer, tanh**, single **linear output neuron WITHOUT bias**
  (deliberately omitted to avoid Jacobian coupling with S_CB). Best size
  **Nh = 5** (they sweep 1–10). Inputs normalized.
- **Feature set ϕ_m (magnetometer-only): Flux A vector components AND the
  uncompensated scalar magnetometer itself** — the scalar used for map-matching
  is also an NN input. (See "endogeneity" note below.)

## Initialization & noise (cold start)

| Quantity | Value |
|---|---|
| β_TL,0 | 0 (18×1) |
| S_CB,0 | 0 |
| NN weights | Glorot, gain γ = 1e-2 (σ_L = γ·sqrt(2/(fanin+fanout))), biases 0 |
| P0,TL | 1e5 · I |
| Q_TL | I |
| P0,NN | I |
| **Q_NN** | **1e-20 · I** (steady-state NN learning rate ≈ frozen) |
| R | 10 nT² |

Warm start = reuse final TL/NN estimates + covariances from a cold-start run.

## Published results (line 1007.06, cold start, Nh = 5)

| Mag | TL+NN | TL-only | Gnadt Online Model 2c [8] |
|---|---:|---:|---:|
| 2 | 51 | 107 | — |
| 3 | 42 | — | 32 |
| 4 | 37 | 58 | 37 |
| 5 | 14 | 15 | 18 |
| 1 (floor) | 17 | 17 | — |

They also acknowledge offline pre-trained methods (Gnadt AIAA [55]) achieve
"comparable or slightly higher" accuracy; their pitch is operational flexibility,
not raw accuracy.

## Deltas from our current reproduction (ekf_tlnn.jl)

1. We use MagNav.jl's Pinson-18 + NN-in-TL-slot; they use a 40+Np error-state
   filter with S_CB, a barometer loop, and vector-mag pseudo-states.
2. We EXCLUDED the scalar magnetometer from the NN inputs (endogeneity collapse
   in our runs); **they INCLUDE it** — the plausible reconciliation is their
   **Q_NN = 1e-20**: after the transient the NN is essentially frozen, so the
   endogenous channel cannot keep absorbing map signal, whereas our weight
   process noise (3e-3) left the network free to chase it. Their χ²-gating
   (off during the first 10 min, on afterwards) points the same direction.
3. We initialize the NN output DC from onboard data; they use S_CB = 0 with
   P0 large and no NN output bias at all.
4. Our activation is swish (MagNav default); theirs is tanh. Our hidden = 8;
   their best = 5.
5. Our R = 144 nT² with map de-trust; theirs R = 10 nT² with gating.

## Open questions for the faithful build

- Exact baro-loop formulation (they cite Gnadt's thesis) — copy from
  MagNav.jl's baro states.
- Whether ϕ_m normalization is causal or whole-line (paper says "all inputs are
  normalized", method unspecified) — assume running/first-window stats, flag as
  an assumption.
- Q_NAV/IMU values (paper's Table for kinematic noise) — to re-extract from
  Section V if needed; our create_model defaults are the fallback, flagged.
