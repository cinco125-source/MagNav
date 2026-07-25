# Red Team Report v3 (post "MAP-spectrum" restructure)

**Target:** `paper/taes_fgo_magnav.tex` — "Joint Aeromagnetic Compensation and
Magnetic-Anomaly Navigation with a Fixed-Lag Factor Graph," IEEE TAES.
**Scope of this pass:** (1) verify v2 Tier-1/Tier-2 fixes survived the Wave 1–2
restructure; (2) hunt for defects the restructure *introduced*; (3) 3-way numeric
audit manuscript ↔ `state.md` ↔ `research/` + `research/gtsam_poc/` result files;
(4) spectrum-frame self-consistency; (5) the four new figures; (6) Abstract/Intro/
Conclusion coherence; (7) standard grep passes.
**Files audited:** `taes_fgo_magnav.tex`, `make_gtsam_figures.py`, `make_figures.py`
(fig_concept/fig_graph), `research/README.md`, `research/gtsam_poc/README.md` +
`gtsam_poc_result_*.txt` + `julia_ref.txt`, `montecarlo_summary.csv`,
`src/fgo_online.jl`, `state.md`.

---

## Accept-probability estimate: ~35% for IEEE TAES (Major Revision)

**Up modestly from v2's 30% — but conditionally.** The restructure genuinely closed
the single most damaging v2 gap (T1-1, zero consistency analysis): §5.5 now carries a
30-seed Monte-Carlo with per-axis 2-DOF ANEES, a χ²₂ NEES band, a ±2σ envelope, and
CI half-widths, and the sliding-window double-counting sub-concern is resolved in code
by a `handoff=:filtered` marginalized boundary (fgo_online.jl L306–310). L_o is now
90 s everywhere (v2 T2-2 gone), the sensor-error over-claim was removed wholesale (v2
T2-5 gone), and the Fig. 1 measurement equation now matches Eq. (3) (v2 T2-3a gone).

**But the restructure manufactured two fresh, reviewer-visible numeric contradictions
and one framing self-contradiction, and left the v2 headline-dependence issue only
half-answered.** The 35% is gated: if the Tier-1 figure↔text realtime-DRMS
contradiction and the Julia-number inconsistency reach a reviewer intact, effective
acceptance is back near 28%. Scoring: TAES base 25% + real flight data 15% +
consistency/NEES now present 10% + ±2σ 5% + recent-lit citations 5% − new Tier-1
fig/text realtime contradiction 10% − internal Julia-number inconsistency 5% − spectrum
"sole variable" self-contradiction + un-tested 8/8 count 5% − residual reimplementation-
dependent headline 5% ≈ 35%.

---

## Tier 1 — Reject levers

### T1-1 (NEW, restructure-induced). Figure 8 (fig:realtime) legend numbers contradict the §5.7 text by ~5× — an undisclosed warm-up-window switch
**Where:** §5.7 text L1455–1457 vs. `make_gtsam_figures.py` `fig_realtime` /
`_load_realtime_series` (L275–300, `drms()` default `warm_s=60.0`), against
`research/gtsam_poc/gtsam_poc_result_prog_lag30.txt`, `..._warm_lag30.txt`,
`..._prog.txt`.
**Problem.** The text states "the 30-second-lag realtime DRMS is **16.5 m** from a cold
TL start and **15.1 m** with a warm prior, close to the **14.8 m** of the 300-second-lag
smoothed estimate." Cross-checking the source files:
- 16.5 m = `prog_lag30` **warm=300 s** (16.53); 15.1 m = `warm_lag30` **warm=300 s** (15.14).
- 14.8 m = `prog`(300 s lag) **warm=60 s** (14.83).
- `fig_realtime` computes every legend DRMS at **warm=60 s**, so its legend reads
  **cold 82.4 m, warm 49.5 m, smoothed 14.8 m** (`prog_lag30`/`warm_lag30` warm=60 s =
  82.35 / 49.54).

So the figure the sentence points to will print "lag=30 s realtime, cold TL (82 m)"
while the text says 16.5 m — a direct, ~5× figure↔text contradiction on a headline
real-time result. The "close to 14.8 m" claim is itself an artifact of mixing warm-up
windows: at a *consistent* warm=300 s the 300 s-lag smoothed is **11.0 m** (`prog`
warm=300 s = 11.01), so realtime 16.5/15.1 is *not* "close" — it is 50 % worse. No
warm-up window is stated anywhere in §5.7.
**Why a reviewer cares.** This is exactly the class of defect a 3-way audit exists to
catch; a legend/text mismatch of this size on the paper's real-time claim reads as
either a fabricated number or an uncontrolled analysis, and it poisons trust in every
other single-run figure.
**Fix.** Pick one warm-up (300 s is the honest choice for a converged-DRMS claim since
the ~4.5-min transient must be excluded), quote all three variants at that warm-up
(realtime cold 16.5 / warm 15.1 / smoothed **11.0**), regenerate `fig_realtime` at the
same warm-up so the legend matches, and state the warm-up explicitly. Note this also
touches §4 L944 (the "near 82 m" transient) and the fig:realtime cross-ref at L945 —
see T2-1.

### T1-2 (carried from v2, half-answered). The headline "beats EKF+TL+NN with no NN" still rests on the authors' own reimplementation of the baseline, and §5.6 does not address it
**Where:** Abstract L78–80; Table III `tab:1007` L1136; Table IV `tab:breadth`
L1242–1249; §5.4 caveat L1220–1226; §5.6 (cross-implementation) L1400–1424.
**Problem.** The accuracy claim is measured against the lineage's reference
implementation "under our tuning ... a faithful, not the authors', filter" (L1222–1224),
whose per-line numbers on lines other than 1007.06 are unpublished and "vary between
runs" (L1224–1226). The new §5.6 cross-implementation subsection looks like a defense
but answers a *different* question: GTSAM reproduces the **proposed FGO's** own result
(implementation independence of *our* method), not the **EKF+TL+NN baseline**. The NN
baseline is still single-run, single-implementation, authors'-own. The reframing to
"match or beat" + foregrounding divergence-avoidance (L1292–1299) is the right move and
partly de-risks this, but a hostile reviewer will still say the few-metre accuracy
margins on well-conditioned Mag 5 (e.g. 18.3 vs 13.1) sit inside the acknowledged
reproduction uncertainty of a baseline that "varies between runs."
**Fix.** State plainly (once, in §5.4) that §5.6 validates the *proposed* estimator's
implementation, not the baseline's, so it does not certify the margin; lean the headline
on the divergence-avoidance + consistency findings (which are robust) and treat the
Mag-5 accuracy margin as secondary. If feasible, add a run-to-run spread (min/max over
seeds) for the EKF+TL+NN column so the margin carries an uncertainty statement.

---

## Tier 2 — Major revision

### T2-1 (NEW). Internal numeric inconsistency: the Julia FGO result for the primary line/sensor is 13.8 m in §5.6 but 13.1/13.5 m in Table III / Table IV
**Where:** §5.6 L1414 ("the Julia smoother's **13.8 m**", full 87-min line 1007.06,
cold-start cabin Mag 5) vs. Table III `tab:1007` L1133–1134 (1007.06 Mag 5 window
**13.5**, window+Huber **13.1**) and Table IV `tab:breadth` L1243 (1007.06 Mag 5 FGO
win **13.1**). Confirmed against sources: `gtsam_poc/julia_ref` = 13.800,
`gtsam_poc/README` = 13.8, `research/README` §2b = 13.8, `state.md` = 13.80 — all the
*cross-impl* leg; but the main-results tables were re-run to 13.1/13.5.
**Problem.** The same physical quantity (Julia FGO, 1007.06, Mag 5, full line, 10-min
warm-up) is reported as three values (13.1, 13.5, 13.8) across the paper. §5.6 pulled its
Julia reference from the older GTSAM-PoC comparison run (13.8) while §5.3/§5.4 use a newer
re-run (13.1/13.5). The cross-impl's whole purpose — "GTSAM 12.4 reproduces the bounded
Julia result" — is undercut when the "bounded Julia result" in the main table is 13.1, not
13.8. A 3-way auditor lands on this immediately.
**Fix.** Reconcile to one Julia number. Either re-run the GTSAM comparison against the
current 13.1/13.5 Julia run, or state in §5.6 that the 13.8 m Julia figure is the
no-Huber window under the legacy handoff used for the port-parity check, and cross-cite
Table III so the reader is not left with two headline numbers.

### T2-2 (NEW). Spectrum frame contradicts itself: "the lag is the sole/single design variable" vs. "two parameters govern the window" and "the two window parameters"
**Where:** Abstract L71–72, §3 intro L500, Contribution 2 L350, and L325 ("the lag is
the single design variable along this spectrum") — all assert a *single* lag knob —
vs. §3.2 L751 ("**Two parameters govern the window**: the length $L_w$ and the overlap
$L_o$") and Conclusion L1490–1491 ("The main tuning burden is **the two window
parameters** and the compensation process noise $\mathbf{Q}^{\beta}$").
**Problem.** The unifying claim of the restructure is that one scalar lag $L$ parameterizes
the whole spectrum. That is exact only for the per-epoch incremental smoother. The window
realization exposes two independent knobs: $L_w$ (adaptation / observability) and $L_o$
(latency = the actual "lag"). The paper never says which of $L_w,L_o$ is "the lag," and the
Conclusion openly lists three tuning parameters. Task item 3's concern is confirmed: the
headline single-variable framing is not self-consistent with §3.2.
**Fix.** Add one clause where the window is introduced: "along the spectrum the *lag*
(look-ahead $L_o$, or a single per-epoch lag) is the latency knob; the window additionally
carries the length $L_w$, which sets adaptation time — see §3.2." Then soften Abstract/
Contribution 2 from "the sole design variable" to "the primary latency knob," and make the
Conclusion's "two window parameters" consistent with it.

### T2-3 (from v2 T1-1, partly open). The count-based 8/8 superiority claim still has no significance test
**Where:** Abstract L76–78; Table IV cell L1251 ("8 / 8"); §5.4 L1181–1184; Conclusion
L1505–1508. §5.5 (the new MC/NEES) is **simulation-only** and does not touch the 8/8 real
claim.
**Problem.** The consistency study certifies covariance credibility in sim, but the
real-data superiority is still asserted as a bare count over 8 paired cases with no test.
As v2 noted, a paired sign test on 8/8 gives p ≈ 0.004 and *helps* the paper; a Wilcoxon
signed-rank on the paired DRMS (with a paired effect size $d_z$) would be stronger and
costs nothing (the paired numbers are already in Table IV).
**Fix.** Add one sentence + one p-value behind the 8/8 claim (sign test or Wilcoxon on the
paired FGO-vs-EKF+TL+NN DRMS); report $d_z$. This converts an easily-attacked count into a
defensible statistic.

### T2-4 (NEW, minor-major). Cross-implementation "agree closely" oversells an 18 % segment gap, and the whole cross-impl rests on one line/one sensor
**Where:** §5.6 L1413–1424; caption fig:crossimpl L1429–1432.
**Problem.** (a) "the two agree closely: ... on the first ten-minute segment 17.5 against
21.4 m" — that is an 18 % difference (GTSAM 18 % *better*), and full-line 12.4 vs 13.8 is
10 % better; "agree closely" is generous, and the consistent direction (GTSAM always
better) invites the question of whether the Julia numbers are optimally tuned. (b) The
cross-impl covers only 1007.06 / Mag 5, so "reproduces the bounded cold-start result"
generalizes one case. (c) `fig_crossimpl` shades the region where the *instantaneous*
GTSAM error dips below the *scalar* Julia DRMS level (21.4), visually implying GTSAM beats
Julia — an apples-to-oranges comparison of a time series against a summary statistic; the
data-provenance note (`fig_crossimpl_note.txt`) itself warns the caption must flag this.
**Fix.** Replace "agree closely" with the quantified deltas ("within 10 % on the full line,
18 % on the segment, GTSAM the lower of the two"); state that the check is on one line/
sensor; in `fig_crossimpl` draw GTSAM's own DRMS level (17.5) as the shading reference
rather than the Julia scalar, or drop the shading.

### T2-5 (from v2 T2-4, partly open). Huber confound in Table IV still lacks a no-Huber column
**Where:** §5.4 L1273–1278 gives one no-Huber example (60.3 m on 1003.02/Mag 4, refined to
43.1 by Huber) and the "window-alone stays bounded on all eight" ablation statement; Table
IV FGO column is Huber-robust while both EKF baselines are not.
**Problem.** Improved over v2 (the survival vs. accuracy split is now argued and one no-Huber
number is shown), but the accuracy half of 8/8 is still not cleanly attributed to structure
vs. kernel, because no no-Huber FGO column appears in Table IV.
**Fix.** Add a no-Huber FGO-window column to Table IV (a one-kwarg rerun, `robust=:none`);
the manuscript already asserts it stays bounded on all eight, so the column strengthens the
structural claim at no narrative cost.

---

## Tier 3 — Polish

- **T3-1. Em-dashes reintroduced.** v2 recorded "no em-dashes"; the Wave 1–2 rewrite
  reinserted 4 parenthetical `---` insertions: L326, L680, L686, L687. Restore the house
  comma/parenthesis style.
- **T3-2. `% TODO` and placeholder biography.** L134 `% TODO: add \cite{trn2025} ...` (source
  comment, won't render but should be resolved before submission); L1610–1614 Bohyun Chang
  bio + `author_placeholder.png` "(Detailed biography to be provided by the author.)".
- **T3-3. Eq. (3) still omits $S_t$** (v2 T3-4 unresolved). Measurement Eq. (3) L445 is
  $z_t=h(\mathbf p_t)+c_t+\eta_t$, but residual Eq. (7) L596 subtracts $S_t$, so the implied
  model $z_t=h+c_t+S_t+\eta$ is never written. Add one clause at Eq. (3) that $S_t$ (the FOGM
  disturbance state) also enters the measurement. (Fig. 1 concept eq is now consistent with
  Eq. (3) — both omit $S_t$ — so v2 T2-3a is fixed; this is only the equation-level gap.)
- **T3-4. Section-letter cross-ref drift.** L656 sends the platform-current absorption check
  to "Section~\ref{sec:results}-C" (Cold-start), but that check is in §5.4 = subsection **D**
  (Breadth), L1279–1288. And L945 sends the "near 82 m" transient to
  "Section~\ref{sec:results}-G" (Real-time, §5.7), but §5.7 reports 16.5/15.1, not 82 — the 82 m
  lives in §4 / fig:warmstart (see T1-1). Fix both targets.
- **T3-5. `research/README.md` is stale — worse than at v2.** Its §2b breadth FGO column
  (32.7/13.8/38.6/14.5/42.6/21.7/26.1/12.4) and §5c (37.0/15.1/32.6/14.2) predate the
  manuscript's re-run (Table IV 30.7/13.1/39.9/15.4/43.1/21.1/28.0/13.1); it still says "5
  lines / 3 flights", "9/10", lists 1006.08 in the breadth table, and its §5c handoff prose
  ("carries the full state + covariance forward ... bounded double-counting") contradicts the
  manuscript's new counted-once claim. Update README so a reviewer with repo access does not
  see the contradiction; note the operative code default is already `handoff=:filtered`.
- **T3-6. ANEES 1.57 is formally *below* the consistency band, not "inside" it.**
  `montecarlo_summary.csv` gives the averaged band [1.984, 2.016]; FGO ANEES 1.572 and EKF
  1.721 are below it (conservative), yet fig:consistency caption L1382 and text L1365–1366 say
  the NEES "stays inside the 95 % χ²₂ band." Per-epoch NEES can sit inside the wide per-epoch
  band while the *average* is significantly below 2; the text is honest ("conservative") but
  the word "consistent/inside" should be "conservative (mildly optimistic-safe)" to avoid a
  reviewer pointing out that ANEES 1.57 fails the two-sided average test.

---

## v2 item-by-item resolution status

| v2 item | Status in v3 | Note |
|---|---|---|
| T1-1 zero consistency/MC, count claims | **Mostly resolved** | §5.5 MC/ANEES/±2σ added; double-count fixed via `:filtered` handoff. **Open:** 8/8 still no significance test (→ v3 T2-3). |
| T1-2 reimplementation-dependent headline | **Partly resolved** | Reframed to "match or beat" + divergence-first; but baseline still authors' single-run reimpl, and §5.6 addresses a different axis (→ v3 T1-2). |
| T2-1 "NN breadth untested" self-contradiction | **Resolved** | Offending sentence gone; only honest "cross-line warm start untested" (L1529) remains. |
| T2-2 L_o = 2 min vs 90 s | **Resolved** | 90 s everywhere; latency 90 s (L1482), cadence 3.5 min (L1484). |
| T2-3 Fig. 1 eq / S_t mislabel / panel(c) | **Resolved (a,b)** | fig_concept eq now $z_t=h+\mathbf A^\top\boldsymbol\beta+\eta$, matches Eq. (3); "aircraft field" label no longer includes $S_t$. Eq-level $S_t$ gap persists (→ v3 T3-3). |
| T2-4 Huber confound in breadth | **Partly resolved** | Survival-vs-accuracy split argued + one no-Huber number; no no-Huber column yet (→ v3 T2-5). |
| T2-5 Contribution 4 factorial/dead-zone over-claim | **Resolved** | Sensor-error study removed from paper; new Contribution 4 is flight-data validation. |
| T3-1..T3-7 (misc) | Mixed | Dead-zone/heading items moot (study removed); README staleness worse (→ T3-5); 2-D Jacobian disclosure retained. |

---

## Numeric 3-way audit (manuscript ↔ state.md ↔ research files)

**Verified consistent:**

| Quantity | Manuscript | Source | ✓ |
|---|---|---|---|
| Cross-impl full line GTSAM / Julia / INS | 12.4 / 13.8 / 317.5 (§5.6) | window_full warm=600 12.41 / julia_ref 13.80 / 317.50 | ✓ |
| Cross-impl segment GTSAM / Julia | 17.5 / 21.4 (§5.6) | window warm=60 17.47 / julia_ref 21.395 | ✓ |
| Realtime speed | 14 ms/epoch (§5.7, Abstract) | prog_lag30.log 83 s / 6000 = 13.8 ms | ✓ |
| MC DRMS + CI | 1.4±0.2 / 4.1±0.4 / 4.9±0.8 (§5.5) | montecarlo_summary 1.39±0.18 / 4.1±0.4 / 4.94±0.75 | ✓ |
| ANEES | 1.57 / 1.72 / 59.8 (§5.5) | montecarlo_summary 1.572 / 1.721 / 59.755 | ✓ |
| Warm-start transient | 82 → ~50 m (§4, fig:warmstart) | prog_lag30/warm_lag30 warm=60 82.35 / 49.54 | ✓ |
| Lock-on ~4.5 min | §4, figs | gtsam_poc README "~4.5-min feature" | ✓ |
| Table III cold-start | 123.7/68.1, 33.0/13.5, 30.7/13.1, 48.8/18.3 | matches §5.3 text and Huber-isolation caption | ✓ (internally) |
| Abstract numeric budget | 14 ms, 8/8 (2 performance numbers) | rule satisfied | ✓ |

**Discrepancies found:**

| # | Quantity | Manuscript values | Root cause | Tier |
|---|---|---|---|---|
| A | Julia FGO 1007.06 Mag 5 full line | **13.8** (§5.6) vs **13.1/13.5** (Table III/IV) | cross-impl used older Julia run; tables re-run | T2-1 |
| B | lag-30 realtime DRMS cold / warm | text **16.5 / 15.1** (warm=300 s) vs fig:realtime legend **82 / 50** (warm=60 s) | undisclosed warm-up switch between text and figure | **T1-1** |
| C | lag-300 smoothed "close to" realtime | text pairs **14.8** (warm=60) with **16.5/15.1** (warm=300); consistent warm=300 value is **11.0** | warm-up mixing to support "close" | T1-1 |
| D | research/README §2b/§5c FGO column | 32.7/13.8/… and 37.0/15.1/32.6/14.2 | README stale vs manuscript re-run | T3-5 |
| E | ANEES vs "inside band" wording | 1.57/1.72 stated "inside 95% band"; band is [1.984, 2.016] | averaged ANEES is below band (conservative), not inside | T3-6 |

**Unverifiable against provided sources (flag, low priority):** the no-Huber 60.3 m and the
41626 m divergence (§5.4 L1275–1276); the current-channel percentages 28–39 % / 2–6 %,
2.6–3.6×, 0.55 correlation (§5.4 L1280–1288, cross-checked only to `research/README` §8a,
which agrees); the "4.1–4.6 m" upper bound in §5.5 L1364 (4.6 not in summary CSV — presumably
the roughened-MPF accuracy).

---

## Grep pass summary

- **Em-dashes:** 4 reintroduced (L326, 680, 686, 687) — T3-1.
- **Placeholders/TODO:** `% TODO` L134; placeholder bio L1610–1614 — T3-2. No `[...]` in body.
- **\textbf{Step N}:** none. **Self-praise lexicon** ("novel", "significantly"): none in
  own-work voice ("state-of-the-art filter" L1269 describes the NN baseline, acceptable).
- **L_o value:** 90 s consistent everywhere (v2 T2-2 clean).
- **Section-letter refs:** two drifted targets (L656 → D not C; L945 → 82 m not in G) — T3-4.

---

## Bottom line for the authors

Two fixes are load-bearing before any reviewer sees this: **(1)** pick one warm-up window
for all §4/§5.6/§5.7 segment numbers and regenerate `fig_realtime`/`fig_warmstart` so their
legends match the text (T1-1 / audit rows B, C); **(2)** reconcile the Julia 1007.06/Mag 5
number to a single value (T2-1 / row A). Both are the direct residue of the restructure
splicing an older cross-impl/real-time result set against freshly re-run main tables, and
both are exactly what a hostile 3-way auditor opens with. After those, add the 8/8
significance test (T2-3) and the no-Huber Table IV column (T2-5) to harden the two
count/accuracy claims, and soften "the sole design variable" to reconcile with "two window
parameters" (T2-2). The consistency study, the divergence-avoidance result, and the
`:filtered` marginalized handoff are genuine, defensible gains over v2.
