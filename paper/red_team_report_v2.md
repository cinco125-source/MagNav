# Red Team Report v2 (Re-review after major revision)

**Target:** `paper/taes_fgo_magnav.tex` — "Factor-Graph Aeromagnetic Compensation for Airborne Magnetic-Anomaly Navigation," resubmitted to IEEE TAES.
**Scope:** verify the prior Tier-1 fixes actually landed, hunt for regressions introduced by (1) the four-lines/two-flights/eight-cases recount, (2) deletion of the standalone Discussion section, (3) the em-dash/emphasis copyedit, (4) the new concept figure (Fig. 1); plus the standard 3-way numeric audit, theorem rigor, code-vs-text, and grep passes.
**Files audited:** `taes_fgo_magnav.tex`, `make_figures.py`, `plot_tracks.py`, `references.bib`, `research/README.md`, `src/fgo.jl`, `src/fgo_online.jl`, `src/fgo_sensor.jl`, `research/fgo_sensor_ablation.jl`.

---

## Accept-probability estimate: ~30% for IEEE TAES (expected decision: Major Revision)

**Scoring rationale.** The revision genuinely repaired the three prior desk-reject-class defects (the Proposition 1 "iff" error, the placeholder `hager2026` bibliography, the divergence-count inconsistency) and expanded the NN baseline from n=1 to all eight breadth cases. That moves this out of "Reject" territory. It is held back from acceptance by one unresolved Tier-1 gap that TAES reviewers weight heavily — **no consistency analysis at all** (no NEES/NIS, no ±2σ, no Monte-Carlo, single-run DRMS) while making count-based superiority claims — and by a headline accuracy claim that rests on an admittedly imperfect reimplementation whose margin is comparable to its own reproduction gap. Several fresh internal contradictions were introduced by the section-folding and the recount (Tier 2). Baseline 25% + real-flight data (+15%) + now-complete Prop.1/proof (+10%) − no MC/NEES/±2σ (−~10%) − reimplementation-dependent headline (−~5%) − residual internal contradictions (−~5%) ≈ 30%.

---

## Tier 1 — Major-credibility / reject levers (ranked)

### T1-1. Zero consistency/statistical analysis, yet count- and margin-based superiority claims
**Where:** whole Results section; `plot_tracks.py fig_poserr` (no covariance channel); Conclusion L1340–1342.
**Problem.** On data with reference truth, the estimator covariance is never validated: no NEES, no NIS, no χ² bounds, no ±2σ error band on `fig_poserr`, and every real-data number is a **single run** (Conclusion L1341 concedes "single-run DRMS on four lines"). The paper nonetheless leans on count statistics — "best of three ... on every full-length line (8 of 8 cases)" (Abstract L52; Table IV cell L1192; Conclusion L1356) — with no significance test (a sign test on 8/8 gives p≈0.004 and would *help* the paper). This gap is compounded by T2-2: the carried prior double-counts the overlap measurements (disclosed at L585–587), which makes the reported `P` optimistic — and the only diagnostic that would expose that (NEES) is absent.
**Why a reviewer cares.** For a TAES estimation paper, filter consistency (NEES/NIS) and error bars are close to mandatory; a hostile reviewer treats their absence plus single-run numbers as "results not shown to be repeatable or statistically meaningful."
**Fix.** Add NEES/NIS with 95% χ² envelopes on ≥2 truth lines (post-processing only — `FILTres.P` is already returned), add ±2σ bands to `fig_poserr`, and either a 20–50-run initial-condition Monte-Carlo or at minimum a paired sign test behind the 8/8 claim.

### T1-2. The headline accuracy claim depends on a reimplementation that underperforms the cited result, with no error bars
**Where:** Abstract L52–54; Table III (`tab:1007`) L1095–1096; Sec. V L876–880; Sec. VI-D caveats L1163–1167, L1216–1218.
**Problem.** The central novelty claim ("reaches the same accuracy without a neural network... beats a reimplemented EKF+TL+NN") is measured against **the authors' own reimplementation**, which is admittedly not the published filter: on Mag 5 the reimplementation reaches 17.5 m versus the published 14 m (L876–880), i.e. +25%, and the paper concedes it lacks "their released architecture / natural-gradient stabilization." The FGO win over the reimplemented NN on the well-conditioned Mag 5 (14.2 vs 17.5, a 3.3 m margin) is *smaller* than the reimplementation's own gap to the published number (3.5 m). Against the actual published values the proposed method is 0.2 m **worse** on Mag 5 (L1053–1054). With no error bars and single runs, "matches or beats" is not safely supported on the well-conditioned sensor.
**Why a reviewer cares.** A superiority claim whose margin is within the acknowledged reproduction error of the baseline is not defensible; this is the exact target of an adversarial reviewer.
**Fix.** Reframe the accuracy claim as "matches the published TL+NN (within 0.2 m) on the well-conditioned sensor and beats it on the ill-conditioned one," foreground that the divergence-avoidance result (not the few-metre accuracy margin) is the robust finding, and add the error bars of T1-1 so the Mag 5 comparison carries a confidence statement.

---

## Tier 2 — Major revision

### T2-1. Internal contradiction: "NN baseline breadth ... remains untested" vs. Table IV, which tests it on all four lines
**Where:** Sec. VI-C L1054–1056 vs. Table IV (`tab:breadth`) L1178–1195 and Sec. VI-D L1130–1131.
**Problem.** L1056 states "the NN baseline breadth across the other four lines remains untested." But the very next subsection's Table IV carries the **reimplemented EKF+TL+NN column for all eight cases** (four lines × two mags), and the text at L1130–1131 explicitly introduces it as "the reimplemented EKF+TL+NN ... the strong causal method that a fair comparison demands." The sentence is a regression: it is a leftover from the single-line framing that the breadth expansion invalidated. As written it flatly contradicts the paper's own table.
**Why a reviewer cares.** A direct self-contradiction between adjacent subsections reads as careless and undermines trust in the rest of the numbers.
**Fix.** Reword to distinguish the two: "the *original authors'* published numbers exist only for line 1007.06; our reimplemented NN is carried across all four lines in Table IV."

### T2-2. Overlap value contradicts itself: L_o = 2 min in two places, 90 s in three
**Where:** L637 ("$L_o=\SI{2}{\minute}$") and L1318 ("the overlap $L_o$, here \SI{2}{\minute}") vs. L922 ("$L_o=\SI{90}{\second}$"), Table I L947 ("$L_o$ ... 90"), and the ratio at L589 ("$L_o/L_w\!=\!0.3$", which forces $L_o=0.3\times300=90$ s).
**Problem.** The look-ahead/commit-latency parameter is given as both 2 min and 90 s. The self-consistent value is 90 s (the 0.3 ratio and Table I agree); L637 and the real-time-latency claim at L1318 ("lags real time by ... 2 min") are therefore wrong, overstating the latency by 33%.
**Why a reviewer cares.** It is a headline reproducibility parameter and a system-latency claim; two of the five statements are simply false, and the latency figure is used to argue operational suitability.
**Fix.** Set $L_o=\SI{90}{\second}$ everywhere; correct the L1318 latency to 90 s.

### T2-3. Concept figure (Fig. 1) measurement equation and S_t description are inconsistent with the model
**Where:** Fig. 1 caption L111–120; `make_figures.py fig_concept` L406, L411–412 (equation `z_t=h+A^Tβ+S_t+η`); vs. Eq. (3) `eq:meas` L362–363 (`z_t=h(p_t)+c_t+η_t`, **no** S_t) and the Nomenclature (S = "absorbs residual magnetic disturbance and map error," a component of the INS state $\mathbf{x}$).
**Problem.** (a) The graphical abstract prints a measurement equation containing $S_t$, but the paper's stated measurement model Eq. (3)/(4) omits $S_t$ — $S_t$ appears only in the *residual* Eq. (7)/(8). So Fig. 1, Eq. (3), and Eq. (8) disagree about whether $S_t$ is in the measurement. (b) The caption L114–115 calls $S_t$ part of "the aircraft field, described by the ... coefficients $\beta_t$ and a disturbance state $S_t$," but the text defines $S_t$ as map/disturbance error, not aircraft field. (c) Panel (c) shows a "causal filter (cold start)" diverging with no indication that the NN causal filter stays bounded — the same "blanket claim that causal filters diverge" the Results section explicitly disowns (L1210–1211).
**Why a reviewer cares.** The graphical abstract is the first thing read; a measurement equation that contradicts Eq. (3) and a physical mislabel of a state are easy, damaging catches.
**Fix.** Make Eq. (3) and Fig. 1 agree — either add $S_t$ to Eq. (3)/(4) (preferred, since the residual uses it) or drop $S_t$ from the figure equation; relabel $S_t$ as "disturbance/map-error state," not aircraft field; and soften panel (c) to "a causal filter *can* diverge" consistent with L1210–1211.

### T2-4. Huber confound disclosed but not isolated in the breadth study
**Where:** Sec. VI-A L916–921; Table IV (FGO column uses Huber per `research/README.md`; EKF baselines do not).
**Problem.** The false "only the estimator structure differs" claim from the prior round is fixed — the paper now states the smoother "additionally applies the Huber kernel" and isolates its effect **on line 1007.06 only** (37.0→32.6 m, L920). But in the breadth table (Table IV) the FGO column is Huber-robust while both EKF baselines are not, and no no-Huber FGO column is shown there. The divergence cases obviously are not a Huber artifact, but the few-metre accuracy margins on the bounded Mag-5 cases could partly be the kernel rather than the structure.
**Why a reviewer cares.** The accuracy half of the 8/8 claim is not cleanly attributed to the factor-graph structure.
**Fix.** Add a no-Huber FGO-window column to Table IV (a one-kwarg rerun, `robust=:none`); if it still wins, the structural claim is strengthened.

### T2-5. Contribution 4 overstates the sensor-error study: "factorial" vs. cumulative, and dead-zone as a "graph variable"
**Where:** Contribution 4 L296–303 vs. Sec. III-F L702–708 and Table VI (`tab:sensor`, cumulative) and L1266.
**Problem.** (a) Contribution 4 claims "a controlled factorial ablation," but Table VI and the script `fgo_sensor_ablation.jl` (L125–131) run a **cumulative single-order** ablation, and L1266 itself concedes "the factors are enabled cumulatively in a single order ... attributes marginal gains only for that order." No factorial (2^k) results appear. (b) Contribution 4 lists "a dead-zone attenuation ... as additional graph variables," but Sec. III-F L702 states the dead-zone term is "**not a graph variable** but a heteroscedastic measurement information," and L706 admits it is "a de-weighting any estimator could apply." So the contribution both mis-names the study and claims as a novel graph variable something that is neither a variable nor unique to the graph.
**Why a reviewer cares.** The prior review flagged exactly this; the body text was made honest but the Contribution bullet still overclaims, creating an internal mismatch.
**Fix.** Change "factorial" to "cumulative" in Contribution 4 (or add the true factorial table to an appendix — the script already supports it); remove the dead-zone from the "graph variables" list and describe it as heteroscedastic weighting.

---

## Tier 3 — Minor / polish

- **T3-1. Dead-zone ε mismatch.** Sec. III-F L703–707 states the estimator weight uses $\epsilon=0.1$; the code default is `dz_floor=0.05` (`fgo_sensor.jl` L20, L111) and the ablation (`fgo_sensor_ablation.jl` L143–145) does not override it, so Table VI was produced with 0.05. (The *injection* used 0.1, L91.) Reconcile the stated ε with the value actually used.
- **T3-2. Heading-coefficient notation clash.** `make_figures.py fig_graph` L195 labels the sensor-error node $\{a_k,b_k\}$ (implying a cos/sin pair), but the model and text use a single cosine coefficient $c_n$ per harmonic (Sec. III-F L693; code comment `fgo_sensor.jl` L131 "not a full Fourier cos/sin pair"). Make the figure use $\{c_n\}$.
- **T3-3. Eq. (9) Jacobian vs. code.** Eq. (9) shows the full $\mathbf{g}_t^\top$, but the online estimator uses only the two horizontal components (`fgo_online.jl` L171 `[Hll[1:2]; ...]`). This is now disclosed in prose at L514–516, but Eq. (9) itself is unannotated; add a one-line footnote at the equation.
- **T3-4. Eq. (3) omits S_t (root of T2-3).** Independently of the figure, Eq. (3)/(4) present the measurement without $S_t$ while the cost residual Eq. (7)/(8) includes it. State once, at Eq. (3), that $S_t$ enters the measurement as the disturbance state.
- **T3-5. Stale provenance in `research/README.md`.** The manuscript is now four lines / two flights / eight cases / 8-of-8, but README still says "5 lines / 3 flights" (§1, §2b), "9/10" (§2b L98–99), lists the dropped 1006.08 row in the breadth table, and §5 thrust-4 still reads "ablation 36.6 → 4.8 m" (the manuscript's Table VI is 42.9→6.2). Manuscript↔figures (`make_figures.py`, `plot_tracks.py`) are consistent; README is the lagging leg of the 3-way audit and should be updated so a reviewer with repo access does not see a contradiction.
- **T3-6. "every full-length navigation line" (Contribution 3 L293).** Only the two Flt1007 lines are free-flight "navigation" lines; 1003.02/1003.08 are survey traverses. The 8/8 win spans all four, so the wording either undersells (only 2 nav lines) or misleads; say "on every full-length line."
- **T3-7. Nomenclature/state-order.** The $[\text{Pinson-17};\beta;S]$ code ordering is now disclosed (L517–519); good. Consider noting it once more at Eq. (1) so the R^18 state and the code permutation are cross-referenced.

---

## What genuinely improved since the prior review (for calibration)

1. **Proposition 1 is now correct and complete.** The observability statement is re-cast as "$[\mathbf{G}\;\boldsymbol{\Psi}]$ full column rank $\iff$ (i) $\ker\mathbf{G}=\{0\}$, (ii) $\ker\boldsymbol{\Psi}=\{0\}$, (iii) trivial range intersection." I verified the three-condition equivalence and the full proof (L762–788): both directions are valid, the flat-map case ($\mathbf{G}\equiv0$) is now consistent with the Remark (no longer declared "observable"), and the new gauge-freedom Remark (`rem:gauge` L805–816) correctly explains the classical-TL rank deficiency as $\ker\boldsymbol{\Psi}\neq\{0\}$ that affects $\beta$-identifiability but not position separability. Corollary 1 is correctly weakened from "observable at any basis dimension" to "**separability** independent of basis dimension." The prior Tier-1 math defect is genuinely repaired.
2. **`hager2026` bibliography is now complete** — full six-author list and title, no "to be finalized" placeholder (`references.bib` L61–71). The prior desk-reject bib risk is gone.
3. **Divergence aggregation is now consistent.** Text (L1134–1135: two lines diverge, a third runs off-map), Table IV (two "div." + one "err." on Mag 4), and README agree for the retained cases; the bogus "1.7×10³ m" range is removed.
4. **Sensor-error ablation is now honest and self-consistent.** Table VI (42.9→18.9→15.5→8.1→6.2) is genuinely monotone in the stated order, matches the script and README §3, attributes gains "only for that order," and adds the model-matched-injection caveat (L1281–1283). The prior "monotonically to 4.8 m" and "largest gains from hard-iron/drift" errors are fixed.
5. **Double-counting is now disclosed, not denied.** The manuscript no longer claims to "prevent double-counting"; L585–591 openly states the overlap is counted twice and bounds it; the `fgo_online.jl` docstring (L243–246) matches.
6. **Huber, 2-D Jacobian, state ordering, IRLS details, and the RTS-vs-QR solver are all disclosed** (L514–519, L663–665, Algorithm 1 L618–619 now "iterated RTS pass," Table I). Algorithm 1 no longer advertises a solver the cold-start results did not use.
7. **NN baseline expanded to the full breadth study** (Table IV carries the reimplemented EKF+TL+NN on all 8 cases), removing the prior n=1 limitation.
8. **A parameter table (Table I) was added** with $\sqrt R$, FOGM $(\sigma,\tau)$, $\mathbf{P}_0$/$\mathbf{Q}^\beta$, Huber $\delta=1.345$, $L_w/L_o/\Delta t$; the $L_w$ selection procedure ("selected on 1007.06, held fixed") is stated.
9. **Recount is internally clean.** Grep confirms no surviving "five"/"ten"/"three flights"/"9 of 10"; abstract, contributions, results, table cell (8/8), captions, and conclusion all read four/two/eight consistently. No em-dashes, no dangling "Discussion" cross-references, no placeholders/TODO, no undefined refs.

---

## Numeric 3-way audit summary
Manuscript ↔ figures ↔ README were checked cell-by-cell. **Manuscript ↔ `make_figures.py`/`plot_tracks.py` match exactly** (Table III ↔ `fig_coldstart` m4/m5; Table IV ↔ `fig_breadth` rows; `fig_winlen` ↔ Table III; Table II 19.4 ↔ text). **README is the only mismatched leg** and is stale in the ways listed in T3-5 (still 5/3/9-of-10, 1006.08 present, "36.6→4.8"). Derived figures re-verified: 51% (28.3→14.0) ✓; 41 NN weights (3×8+8+8+1) ✓; "4 to 76 m" NN margins ✓; 8/8 bold cells ✓; 86% ablation reduction (42.9→6.2) ✓; 4.4 m / 0.2 m published deltas ✓.
