# Restart audit — 2026-08-19

Written in a remote web session where **nothing could be executed**: Julia
installation is blocked by the egress policy (`julialang-s3.julialang.org` → 403)
and the flight data is blocked as well (`dropbox.com` → 403). Everything below
comes from reading the committed code. **No number in this file was produced by a
run**; each item states how to settle it locally.

---

## 1. Why the code drifted: the feedback loop was CI

`.github/workflows/` holds 27 workflows, one per experiment
(`mpf_np.yml`, `mpf_np2.yml`, `mpf_np_t05.yml`, `fgo_overlap.yml`,
`fgo_winlen.yml`, …). `fgo_research.yml` states its own cost in a header
comment: *"This keeps the push cycle at ~1 h instead of ~3 h."*

So the development loop was: write blind → push → wait an hour → read a log. At
that latency nobody, model or human, probes an assumption; they infer it. Every
defect below is a symptom of that loop rather than of any single mistake.

**Fix: move development to a local Julia + Python environment (see §6) and keep CI for
confirmation only.**

## 2. The manuscript's engine is not the repository's code

The TAES headline results — eight cold-start cases, causal and smoothed
estimates, 300 s lag, 1 Hz states, per-update timings — are produced by
`research/gtsam_poc/run_gtsam_decimated.py` (Python + GTSAM `IncrementalFixedLagSmoother`),
driven by `research/gtsam_poc/breadth_final.sh`, which runs from a local Windows
path (`cd /mnt/c/Users/cin64/Desktop/MagNav`, `PY=~/gtsam_env/bin/python`) against
`~/magnav_data/`.

The Julia `src/fgo_online.jl` sliding window appears only as a reference column;
`research/gtsam_poc/collect_results.py` documents it as *"the Julia sliding-window
reference"*.

Consequences to resolve before any further work:
- Two implementations of the same estimator exist and **neither validates the
  other**. They differ in more than language: the Julia path is a per-window batch
  smoother with a hand-rolled prior handoff, the Python path is a per-epoch
  incremental smoother.
- The manuscript names neither, so a reader cannot tell which produced a number.
- The paper's results are reproducible only on one machine, from shell scripts
  with absolute local paths, with outputs stored as `.txt` logs.

**Decide which implementation is canonical, then make the other reproduce it.**

## 3. Defect (confirmed by reading): the window handoff skips one propagation

`src/fgo_online.jl:300-320`

```julia
gc1 = (i1==N) ? N : min(i0+stride-1, N)   # last committed epoch, global index
...
if filt
    x0_pr = res[2][:,lc1]                  # FILTERED marginal AT epoch gc1
    P0_c  = (res[3][:,:,lc1] .+ res[3][:,:,lc1]') ./ 2
...
i0 += stride                               # next window starts at gc1+1
```

`fgo_rts_pass` consumes `x0`/`P0` as the **predicted (a priori) distribution of
the window's first epoch** — its forward pass applies the measurement update
immediately, before any propagation. The handoff instead supplies the
**posterior at the previous epoch** `gc1`, and the next window begins at `gc1+1`.
The transition `Φ_{gc1}` and the process noise `Qd` for that one step are never
applied, so each window starts from a prior that is both time-shifted by one
sample and over-confident by `Qd`.

Magnitude is presumably small at `dt = 0.1 s` and one handoff per `stride`, but it
is wrong exactly where the manuscript claims a "double-count-free marginalized
handoff", which is the mechanism the fixed-lag contribution rests on.

Fix: `x0_pr = Φ_gc1 * x_filt[:,lc1]`, `P0_c = Φ_gc1 * P_filt[:,:,lc1] * Φ_gc1' + Qd`.
`Φ` for that epoch is already available as `Phi_a[:,:,gc1]` inside the window solve
and must be returned, or recomputed with `get_Phi` at index `gc1`.

Settle it with V2 below: the bug makes windowed ≠ batch by exactly the missing
propagation, so the test both proves the defect and validates the fix.

## 4. Open question (not verified): does the Julia "causal" estimate deserve the name

`fgo_online` returns `x_filt` from the forward pass of its **final** Gauss-Newton
iteration — a pass linearized about the smoothed trajectory of the whole window.
It has therefore already used future measurements through the linearization point
and is not a causal estimate in the sense the manuscript claims ("a causal estimate
with no look-ahead", abstract; §IV "the value of the newest variable").

If every causal number in the paper came from the GTSAM per-epoch run, this is
harmless and only the Julia docstring needs correcting. If any causal number came
from the Julia `x_filt`, the like-for-like comparison against the EKF baselines
does not hold.

**Settle it by tracing every causal figure in `paper/taes_incremental.tex` back to
the file that produced it.** Not answerable from the code alone.

## 5. Checked and clean

The state layout in `fgo_online` — Tolles-Lawson coefficients at `18:17+nx_TL`
with the FOGM state last, and `H = [Hll[1:2]; zeros(nx-3-nx_TL); A[t,:]; 1]` —
matches upstream `src/ekf_online.jl:79,125` exactly. Initially suspected, then
ruled out.

## 6. Verification ladder (build this before any new result)

Ordered so each rung is meaningless without the one below it. Each has a pass
criterion that does not mention DRMS: these test whether the implementation is
correct, not whether the answer is good.

**V1 — batch equivalence.** On a linear-Gaussian problem (fixed linearization
point, no map nonlinearity), `fgo(solver=:gn)`, `fgo(solver=:rts)` and a plain
Kalman filter followed by an RTS pass must agree.
*Pass:* max state difference below solver tolerance, not "close".

**V2 — window equals batch.** With `win` covering the whole record, and again
with a short `win` on a linear-Gaussian problem, `fgo_online_window` must
reproduce the single-batch `fgo_online` estimate. A correct marginalized handoff
loses nothing on a chain.
*Pass:* identical to solver tolerance. **Expected to fail today** — that is §3.

**V3 — injected-truth recovery.** Simulate with known Tolles-Lawson coefficients
and a known trajectory; the joint estimate must recover both.
*Pass:* coefficient error inside the reported covariance; state error inside ±2σ.

**V4 — covariance honesty.** Repeated realizations, ANEES for the 2-DOF
horizontal position against the χ²₂ band.
*Pass:* inside the band, or a written explanation of the direction of the bias.

V1–V3 need no flight data — a simulated map and trajectory suffice, so they run
locally in seconds and belong in `test/test_fgo.jl`, not in CI-only scripts.

## 7. Local bootstrap

```bash
# in WSL
curl -fsSL https://install.julialang.org | sh          # juliaup
git clone <this repo> && cd MagNav
git checkout claude/magnav-research-restart-hrllr4
julia --project=. -e 'using Pkg; Pkg.instantiate()'    # first run precompiles
julia --project=. -e 'using MagNav; println(pathof(MagNav))'   # smoke test
julia --project=. -e 'using TestItemRunner; @run_package_tests filter=ti->occursin("fgo",ti.name)'
```

The GTSAM side already exists at `~/gtsam_env` with data in `~/magnav_data/`.
`research/gtsam_poc/export_full_line.py` rebuilds every input (Pinson Φ, TL A,
INS/truth, map grid) from the raw SGL HDF5 without Julia, and self-validates
against the Julia export.

## 8. First task, in order

1. Stand up the local environment (§7) and run the existing `fgo` tests. Report
   what actually passes — the current state is unknown, not assumed green.
2. Write V1 and V2 as tests. V2 is expected to fail; confirm the failure before
   fixing anything.
3. Fix §3, show V2 turning green, and re-run one real line to measure whether the
   headline numbers move.
4. Answer §4 by tracing the manuscript's causal numbers to their source files.
5. Only then decide which implementation is canonical (§2).

Nothing in `paper/` gets edited until steps 1-4 are done.
