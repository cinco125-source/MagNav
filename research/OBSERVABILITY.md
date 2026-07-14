# Observability of cold-start joint aeromagnetic compensation + navigation

Working notes toward the algorithmic contribution. All numbers are from real SGL
2020 data, line 1007.06 (full 87 min, uncompensated cabin Mag 4), produced by the
CI scripts `research/observability.jl` (geometry) and `research/observability_ekf.jl`
(controlled filter runs).

## The question

When we estimate aircraft-interference compensation *jointly* with position from a
scalar magnetometer matched against a map, what governs whether the estimator
converges or diverges from a cold start?

## What we found (and what we had to unlearn)

### Two distinct failure modes, and a sweet spot

Sweeping the Tolles-Lawson compensation basis in the **linear online-TL EKF**
(`ekf_online`, no neural network), holding everything else fixed and only changing
the TL `terms`:

| compensation basis | # TL states | global ρ²(map·basis) | DRMS (>10 min) |
|---|---:|---:|---:|
| permanent (3) + bias | 4 | 0.019 | **7631 m — diverged** |
| perm + induced (9) + bias | 10 | 0.845 | **46.9 m** |
| perm + induced + eddy (18) + bias | 19 | 0.850 | **46.7 m** |

and, from the reproduced online **EKF+TL+NN** (`research/paper_impl.jl`):

| compensation | features | DRMS |
|---|---|---:|
| NN on permanent TL features | exogenous | **40.0 m** |
| NN on permanent TL + **mag_uc** | **endogenous** | **5813 m — collapsed** |

DRMS is **U-shaped** in compensation expressiveness, bounded by two failures:

1. **Under-compensation** — the basis is too weak to represent the real
   interference (permanent-only cannot model the induced/eddy attitude terms that
   dominate a cabin magnetometer). The uncompensated residual is then attributed
   to position, which diverges (7631 m). *This has nothing to do with
   observability.*
2. **Observability collapse** — the basis can reproduce the map signal we
   navigate on, so the compensation absorbs it and position drifts while the
   residual stays small (5813 m at `max|resid|` 78 nT).

Between them is a broad sweet spot (40–47 m, at the paper's cold-start band).

### A prediction we made — and falsified

We first hypothesised that the collapse index **global ρ²** (the fraction of the
map anomaly a basis can reproduce) would *monotonically* predict DRMS. The
controlled sweep **falsifies** this: the lowest-ρ² basis (permanent, ρ²=0.019)
had the **worst** DRMS (under-compensation), and two bases with nearly identical
global ρ² (≈0.85) behaved oppositely in different estimators — the 18-column
*attitude* basis converged (46.7 m) while the *mag_uc*-augmented basis collapsed
(5813 m). **Static fit-capacity ρ² is not the right invariant.**

### The right invariant: feature endogeneity

What separates the safe high-ρ² basis (attitude) from the unsafe one (mag_uc) is
not how well it fits the map, but **whether the feature responds to a position
error**:

- The measurement is `z = h_map(p) + comp(features) + noise`. A position error
  `δp` perturbs the measurement by `∇h_map · δp`.
- `mag_uc` **contains** `h_map(p)` — it is a function of position — so
  `comp(mag_uc)` can absorb `∇h_map · δp` directly. Position and compensation are
  confounded ⇒ **collapse**.
- Attitude / fluxgate features are functions of aircraft **orientation**, which is
  exogenous to `δp` (a position error does not rotate the aircraft). They may
  correlate with the map incidentally over a given flight (hence high global ρ²),
  but that correlation is fixed by the trajectory and **does not respond to
  `δp`**, so it cannot absorb `∇h_map · δp`. Observable at **any** expressiveness.

So the map-leak / observability-collapse hazard is governed by the **endogeneity**
of the compensation features (their coupling to position), not by their static
map-fitting capacity or their dimension. `ρ²(map·basis)` conflates the two; a
correct index must measure the *position-responsive* part of that coupling.

## The design principle (contribution)

> Cold-start joint aeromagnetic compensation and navigation has an
> expressiveness sweet spot bounded by **under-compensation** (basis too weak)
> and **observability collapse** (basis endogenous to position). Compensate with
> the **most expressive model drawn from exogenous (position-independent)
> features** — attitude / fluxgate, never the scalar being navigated on.

This reframes the reproduced result: our sliding-window FGO-online and the
paper's stabilized NN both live in the sweet spot because both compensate from
exogenous features; the "remove `mag_uc`" fix is the special case of enforcing
exogeneity.

## Open / next

- **Endogeneity index.** Replace `ρ²(map·basis)` with a measure of the
  *position-responsive* coupling — e.g. the alignment between the compensation
  Jacobian `∂comp/∂(features)·∂(features)/∂p` and the map gradient `∇h_map`, or,
  operationally, the correlation of each feature with `h_map(p)` **after removing
  the part explained by exogenous trajectory variables**. Exogenous features
  should score ≈0; `mag_uc` should score high.
- **Under-compensation index.** Quantify the weak-basis failure (residual power
  the basis cannot represent) so the sweet spot can be predicted from both sides.
- **Online gate.** If any feature's endogeneity index rises during flight,
  down-weight its adaptation — the adaptive analog of the exogeneity rule.
- **CRLB / identifiability.** Formalise the two boundaries as an identifiability
  condition for the joint (compensation, position) estimate.
