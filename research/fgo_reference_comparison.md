# Our FGO against an independent one

Structural audit against `BOBI9710/FGO-Space-based-Target-Tracking` (MATLAB,
stereo space-based missile tracking). Different problem entirely -- 9-state
target in ECI, stereo camera pixel measurements, Keplerian two-body plus Singer
acceleration -- but the same estimator shape, written by someone else, with a
hand-rolled solver rather than GTSAM. That makes it a useful check on whether
ours is doing the standard thing.

## Verdict

Every essential element matches. Nothing in ours is wrong.

| | theirs (`FGO.m`) | ours (`run_gtsam_decimated.py`) |
|---|---|---|
| variables | 9 per node (r, v, u), one node per epoch | 37 per node, one node per epoch |
| window | `Nnode = 9` nodes | 300 states at a 300 s lag |
| prior | `PriorR`/`PriorD`, square-root form, carried | `PriorFactorVector(X(0), 0, P0)` then marginalized |
| dynamics factor | binary, `resDyn = Xnow - PropagateState(Xpast)` | binary, `e = xb - Phi @ xa` |
| measurement factor | unary per node, stereo pixel | unary per node, map + TL + FOGM |
| whitening | `Wnow = chol(Rtotal,'lower')`, `A = W \ J` | GTSAM noise model, verified `W0'W0 = P0^-1` |
| solve | `qr(A,0)`, `dx = -R \ d` | iSAM2, QR factorization |
| marginalization | explicit Schur complement on the oldest node, eigendecomposed to a new square-root prior | iSAM2 Bayes tree, keys past the lag |
| relinearization | every factor in the window re-linearized every step | threshold-based, selective |

The whitening convention is the same one we verified: `A = W \ J` with `W` a
lower Cholesky factor gives `A'A = J' R^-1 J`. Their `MarginalizeOldestNode`
forms `H = A'A`, Schur-complements the oldest 9 states out, eigendecomposes and
truncates to a square root -- which is what the Bayes tree does for us
incrementally. Their `b_marg = b_marg + A_marg*delx` shifts the residual to the
updated linearization point before marginalizing; GTSAM does the equivalent
internally.

So the differences are implementation, not formulation. Ours is the library
version of what they hand-rolled.

## The finding that matters

Their FGO reports the OLDEST node in the window:

```matlab
saveIdx = i - Nnode + 1;
Xsave(:,saveIdx) = Xupdated(1:Nx);     % node 1 = oldest = smoothed, lag 8
```

Their EKF reports the filtered state at the current epoch:

```matlab
[X, P, ~, nis, ndof] = Correction(...);
Xsave(:,i) = X;                        % causal
```

Both land in `PARAM_SIM.OUT_MIS_est` and are compared. **That is a lag-8
smoothed estimate against a causal filter** -- the comparison our Contribution 3
says the literature rarely distinguishes. Here is a concrete instance, in an
independent codebase, on a different problem.

This is worth stating plainly because of what we measured on our own data. Our
causal column ties the EKF (0.970 at the benchmark's 0.1 m initial position,
0.997 at 100 m) while our smoothed column reads 0.250 to 0.063 depending on the
regime. If we reported only the smoothed number -- as this repository does --
our result would look like a factor of four to sixteen and nobody would ask.
We report both, found the causal margin is not real, and spent a day on it.

That is the right call and it should stay. But it does mean the comparison in
Contribution 3 is not a rhetorical point: it changes the headline number by an
order of magnitude, and there is now a citable example of the practice.

## One thing worth borrowing

They estimate the Singer correlation constant online. `AdvanceTauModeFilter`
runs a multiple-model filter over a discrete set of alpha values, carrying mode
probabilities and reporting `1/alpha_bar` as the effective correlation time.

We fix the FOGM correlation time at 180 s -- and `noise_model_audit.py` measured
the actual value on line 1007.06 Mag 5 at 35 s, with the measured correlation
falling to 0.03 by 60 s where `exp(-60/180)` is still 0.72. Retuning to the
measured value bought 4.8%. A mode filter over tau is the principled version of
that retune, and it is the one idea in this repository that transfers directly.

Their disturbance state is doing real work (it is the target's unmodelled
acceleration); ours reads 0.11 nT rms and is doing essentially nothing, so the
gain here is likely small. Worth an hour, not a week.

## What does not transfer

- Their window is 9 nodes. Ours is 300. Theirs re-linearizes everything every
  step, which is cheap at 9 nodes and is why they need no relinearization
  threshold; at 300 states that is what `--full-relin` costs.
- Their measurement is a 2-vector per camera with a genuinely nonlinear
  projection through `Kcam` and a moving observer. Ours is a scalar map lookup
  that is nearly linear over the position error. That difference is the whole
  reason relinearization pays for them and buys us 3%.
- They have no compensation nuisance at all. Our 19 Tolles-Lawson states
  competing with 2 position states inside one scalar residual has no analogue
  here.
