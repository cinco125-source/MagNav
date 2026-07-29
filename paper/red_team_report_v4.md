# Red Team Report v4 (post v3-prescription application)

**Target:** `paper/taes_fgo_magnav.tex` — "Joint Aeromagnetic Compensation and
Magnetic-Anomaly Navigation with a Fixed-Lag Factor Graph," IEEE TAES.
**Commits under review:** 13d2dc9 (v3 fixes) + 27c6a34 (proofread).
**Scope of this pass:** (1) item-by-item verification that every v3 prescription
actually landed, with an independent recompute of `drms300_*` against the CI result
files; (2) hunt for 2nd-order defects the fixes introduced; (3) fresh-eyes full
re-audit (Nomenclature/Eq consistency, Appendix, four-vs-three framing, cross-run
provenance); (4) accept-probability re-estimate and a submit-gating punch list.
**Files audited:** `taes_fgo_magnav.tex` (full), `make_gtsam_figures.py`,
`red_team_report_v3.md`, `state.md`, `research/gtsam_poc/README.md` +
`gtsam_poc_result_{prog,prog_lag30,warm_lag30,window,window_full,batch}.txt` +
`julia_ref.txt`, `research/README.md`.

---

## Accept-probability estimate: ~40% for IEEE TAES (Major Revision, favorable half)

**Up from v3's gated 35%.** The two load-bearing v3 Tier-1 levers are genuinely
closed and no new Tier-1 was introduced:

- **T1-1 (fig↔text realtime ~5× contradiction) — RESOLVED and independently
  reverified.** `fig_realtime` now labels every DRMS with `drms300_*`
  (`make_gtsam_figures.py` L342–348, `warm_s=300.0` L301–305) and prints a title
  disclosing the convention (L349). I recomputed the four legend numbers straight
  from the CI result files: cold **16.53**, warm **15.14**, smoothed(l300) **11.01**,
  INS **52.00** at `warm=300 s` — matching §5.7 text 16.5 / 15.1 / 11.0
  (L1463–1464) to the digit. The transient story is quarantined in `fig_warmstart`
  under the `warm=60 s` convention (82.35 / 49.54), matching §4/§5.7's 82.4 / 49.5
  (L1460–1461). The dual-convention split is now explicit in both figures and both
  text sites. Clean.
- **Julia 13.8/13.1 internal inconsistency — resolved *inside the manuscript*.**
  Every occurrence in the .tex is now 13.1 (Table III L1135, Table IV L1244/L1250,
  §5.6 L1418 with a `Table~\ref{tab:breadth}` cross-ref). A reader without repo
  access sees one number. (But see T2-1 below — the fix relocated the inconsistency
  to the artifacts and spliced two Julia runs inside §5.6.)

Scoring: TAES base 25% + real flight data 15% + NEES/consistency present 10% + ±2σ
plot 5% + Prop 1/Lemma 1 with full proofs 10% + recent-lit (hager2026, angappan2026,
lathrop2024, qctrl2025) 5% − residual reimplementation-dependent accuracy headline
(T1-2) 10% − §5.6 cross-run Julia splice + artifact mismatch (T2-1) 5% − no-Huber
column still absent / MC only 30 seeds 5% − four-vs-three + c_t definitional slips
(T3) 5% ≈ **40%**. No longer gated on a figure contradiction.

---

## v3 item-by-item landing verification

| v3 item | Verdict | Evidence |
|---|---|---|
| **T1-1** fig↔text realtime ~5× | **LANDED** | `drms300_*` legend = text 16.5/15.1/11.0; recomputed from `*_result_prog_lag30/warm_lag30/prog.txt` warm=300 rows. Figure title L349 states convention. |
| **T1-2** headline rests on own NN reimpl | **Standing** (structural) | Still single-run authors' reimpl (§5.2 L1067–1068, §5.4 L1221–1227). Hedged to "match or beat" + divergence-first, but not closed. → carried to T2 below. |
| **T2-1** Julia 13.1/13.5/13.8 three-way | **Half-landed** | .tex now uniformly 13.1; but §5.6 now splices two Julia runs and artifacts still say 13.8 → **reopened as T2-1'** below. |
| **T2-2** "sole/single design variable" | **LANDED** | 4/4 sites now "primary design variable" (L71, L325, L350, L501). `grep` for "sole/single design" returns only the legitimate "sole source of horizontal position" (L457). Conclusion "two window parameters" (L1500–1501) now compatible with "primary" (not "sole"). Minor residual: paper still never writes "the lag ≡ L_o for the window." |
| **T2-3** 8/8 has no significance test | **LANDED** | §5.4 L1294–1296: paired sign test, one-sided 2^{-8}≈0.004. Verified 2^{-8}=0.00390625; "one-sided/paired sign test" naming correct. Residual: no `d_z`, no Wilcoxon (v3 also asked); and the p-value is *pairwise* while the surrounding sentence says "best of the four" (see T3-2). |
| **T2-4** crossimpl "agree closely" oversells | **Mostly landed** | "agree closely" → "land within a metre" (L1416) with numbers exposed (17.5 vs 21.4, L1418–1419); shading removed (`make_gtsam_figures.py` L222–223); caption flags scalar-reference (L1433–1435); one-line/one-sensor scope stated (L1424–1427). Residual: "slightly the better of the two" (L1419) still understates the 18% segment gap. |
| **T2-5** no-Huber FGO column in Table IV | **OPEN** | Table IV (L1238–1254) still Huber-only; one inline no-Huber number (60.3 m, L1276). `state.md` confirms this is deferred to a CI/remote Julia rerun. |
| **T3-1** em-dashes | **LANDED** | `grep -- "---"` returns nothing. |
| **T3-2** TODO + placeholder bio | **OPEN** | `% TODO trn2025` L134; `author_placeholder.png` + "(Detailed biography to be provided...)" L1620–1624. Pre-submission cleanup. |
| **T3-3** Eq (3) omits S_t | **Landed but introduced T2-2' (c_t defect)** | Clause added at L450 ("c_t ... decomposed ... into a TL part and a FOGM remainder S_t"), which now **contradicts** the c_t=A^Tβ definitions at Nomenclature L102 and Eq (5) L473. See T2-2' below. |
| **T3-4** section-letter drift (L656→C, L945→G) | **LANDED** (those two) | L657 now →`sec:results}-D` (platform-current check is in §5.4=D ✓); L946 →`-G`, and §5.7 (G) now actually reports 82.4 ✓. **But a new drift appeared at L1352** (see T3-3 below). |
| **T3-5** research/README stale | **OPEN (worse-confirmed)** | README still shows FGO 32.7/13.8 (L99–100), "5 lines/3 flights" (L28,L77), "9/10 diverge" (L292), win-5min 37.0/15.1 (L250). Contradicts Table III/IV. Repo-only exposure. |
| **T3-6** ANEES 1.57 "inside band" vs below | **LANDED** | Text now separates per-epoch NEES "inside the 95% χ²₂ band" (L1368) from averaged ANEES "below the ideal 2 ... conservative" (L1357–1359). Honest. |

---

## Tier 1 — Reject levers

**None newly introduced, and both v3 Tier-1 levers are closed.** This is the
headline improvement of this revision: the manuscript no longer carries a
figure↔text numeric contradiction, and no single defect now rises to desk-reject
level. The strongest standing risk (T1-2, baseline provenance) has been demoted to
Tier 2 by the "match or beat" + divergence-first reframing, and is repeated below at
its true severity.

---

## Tier 2 — Major revision

### T2-1' (NEW, introduced by the v3 fix). §5.6 now splices two different Julia runs, and the cross-implementation artifacts still contradict the manuscript
**Where:** §5.6 L1416–1421; against `research/gtsam_poc/julia_ref.txt` (13.800 /
21.395), `gtsam_poc/README.md` headline table (Julia 13.8), `make_gtsam_figures.py`
`JULIA_FULLLINE_DRMS = 13.8` (L46), `state.md` core-numbers table (13.80).
**Problem.** To kill the v3 three-value inconsistency the authors changed the §5.6
full-line Julia figure from 13.8 to **13.1** and cross-cited Table IV. But §5.6's
*segment* comparison is still **17.5 vs 21.4** (L1418), and 21.4 comes from
`julia_ref.txt`, i.e. the *same cross-impl session whose full-line Julia value is
13.8*. So §5.6 now pairs a full-line Julia number from the **main-table re-run
(13.1, Huber-robust)** with a segment Julia number from the **legacy cross-impl run
(21.4 / 13.8-family)**. The two halves of the cross-implementation subsection no
longer come from one Julia run. Worse, the GTSAM 12.4 m was bit-exact-validated
against the *13.8* Julia export (README "port validated bit-exact"; `julia_ref`),
not against 13.1 — so "12.4 vs 13.1, GTSAM slightly better" is not the apples-to-
apples pairing the subsection claims to be. Table III shows 13.1 is the *Huber*
window (no-Huber window = 13.5); it is not established that the GTSAM window uses the
same kernel setting, so the direction-of-difference claim ("GTSAM the lower of the
two", L1419) may be a kernel artifact rather than an implementation one.
**Why a reviewer cares.** The entire value of §5.6 is "same problem, independent
code, same answer." Mixing a 13.1 (new-run, Huber) full-line against a 21.4
(legacy-run) segment, while the repo's `julia_ref`/README say 13.8, is exactly what a
3-way auditor with repo access opens with — and it re-creates, one layer down, the
inconsistency v3 tried to remove.
**Fix (pick one, and make artifacts agree):** (a) **Honest same-run pairing** —
report §5.6 full-line as **12.4 vs 13.8** (the cross-impl's own Julia reference) and
add one clause "the current Huber-robust window is 13.1 m (Table IV); the 13.8 m here
is the legacy no-Huber window run used for the bit-exact port check." This keeps
§5.6 self-consistent (both halves from `julia_ref`) and honest. (b) Re-run the GTSAM
window against the current 13.1 Julia config (matched kernel) and regenerate
`julia_ref.txt`/README/`JULIA_FULLLINE_DRMS`/`state.md` to 13.1. Either way,
confirm the GTSAM and Julia windows use the **same Huber setting** before claiming a
direction.

### T2-2' (NEW, introduced by the T3-3 fix). c_t is now defined two incompatible ways: A^Tβ (Nomenclature, Eq 5) vs A^Tβ + S_t (new L450 clause)
**Where:** Nomenclature L102 (`c_t = A_t^T β_t`); Eq (5) L473 (`c_t = A_t^T β_t`);
vs the new clause at L450 ("c_t is the aircraft interference, decomposed ... into a
Tolles–Lawson part and a first-order Gauss–Markov remainder S_t"); residual Eq (7)
L597 (`r_t = z_t − h − A^Tβ − S_t`).
**Problem.** The v3 T3-3 fix correctly made the *measurement* include S_t, but it did
so by redefining c_t at L450 as (TL + S_t), while leaving c_t = A^Tβ at both the
Nomenclature and Eq (5). The result: the single symbol c_t means "TL only" in two
places and "TL + FOGM" in a third. The math in Eq (7) is correct
(z = h + A^Tβ + S_t + η); only the naming is now self-contradictory, and it sits in
the foundational measurement model where TAES reviewers read most carefully.
**Fix (one line).** Keep c_t = A^Tβ (TL only) everywhere and instead write Eq (3) as
`z_t = h(p_t) + c_t + S_t + η_t`, deleting the "decomposed into ... S_t" clause at
L450 (or replacing it with "c_t is the Tolles–Lawson interference of Eq (5); a
first-order Gauss–Markov remainder S_t enters additively, see Eq (7)"). This
simultaneously closes the original T3-3 gap and removes the new overload.

### T2-3 (carried, T1-2). Accuracy headline still rests on a single-run, authors'-own reimplementation of the EKF+TL+NN baseline
**Where:** Abstract L78–79; §5.2 L1067–1068; §5.4 caveat L1221–1227; Table IV
L1240–1250.
**Problem.** Unchanged in substance since v3: the per-line NN numbers are "a
faithful, not the authors', filter" whose "per-line DRMS varies between runs"
(L1224–1226). §5.6 validates the *proposed* estimator's implementation independence,
not the baseline's, so it does not certify the margin. On well-conditioned Mag 5 the
margins (18.3 vs 13.1; 20.7 vs 13.1; 30.4 vs 15.4) are real but sit near a baseline
that the authors themselves say varies run-to-run.
**Fix.** Add a run-to-run spread (min/max or ±σ over a few seeds) for the EKF+TL+NN
Table IV column so the margin carries an uncertainty; and state once (in §5.4) that
§5.6 certifies the proposed side only. Lean the Abstract on the robust findings
(bounded 8/8, divergence avoidance, consistency) and treat the accuracy margin as
secondary. Low cost, converts the most-attacked claim into a defended one.

### T2-4 (carried). No-Huber FGO column absent from Table IV
**Where:** Table IV L1238–1254; inline example L1276 (60.3 m).
**Problem.** The accuracy half of 8/8 is still not cleanly split between structure
and kernel because Table IV's FGO column is Huber-robust while both EKF columns are
not. The manuscript already asserts the window stays bounded on all eight without
Huber.
**Fix.** Add a `robust=:none` FGO-window column (one-kwarg rerun). `state.md` flags
this as the remaining CI task; it strengthens the structural claim at no narrative
cost.

---

## Tier 3 — Polish

- **T3-1 (NEW, fresh-eyes). "four" vs "three" estimators flip-flops, and §5.4
  contradicts itself.** Abstract "most accurate of **four** estimators ... 8 of 8"
  (L75–76), Contribution 4 "most accurate of **four**" (L366), and §5.4 L1294 "best
  of the **four** on all eight" — but Table IV caption says "best of **three**"
  (L1232), the 8/8 cell is over three tabulated methods (L1252), and §5.4 L1183 says
  "best of the **three** on all eight." So §5.4 asserts both three (L1183) and four
  (L1294) within one subsection. The "fourth" is MPF+TL, which is reported separately
  and excluded from Table IV / the 8/8 count. Also the sign-test p=2^{-8} at L1295 is
  a *pairwise* (2-way) statistic but is attached to the "best of the four" sentence.
  **Fix.** Pick one framing. Cleanest: "best of the three tabulated estimators on all
  8 cases (the marginalized particle filter, reported separately, is never
  competitive)"; keep 2^{-8} explicitly as the FGO-vs-EKF+TL+NN pairwise sign test.
  Make Abstract/Contribution/§5.4 all say the same count.

- **T3-2 (NEW). Section-letter drift at L1352.** §5.5 (subsection E) says "The larger
  cold-start separations are in Sections~\ref{sec:results}-**B** and~-**C**." The
  cold-start accuracy separations are in §5.3 (C, primary line) and §5.4 (D, breadth);
  §5.2 (B) is Baselines and reports no separation. Should read "-C and -D."

- **T3-3. `% TODO` and placeholder bio unresolved.** L134 `% TODO: add \cite{trn2025}`;
  L1620–1624 Bohyun Chang placeholder bio + `author_placeholder.png`. (Template
  fields L19 `\jmonth{XXXXX}`, L46 received-date are normal pre-submission stubs.)

- **T3-4. research/README.md still stale (repo-only).** FGO 32.7/13.8 (L99–100) vs
  Table IV 30.7/13.1; "5 lines/3 flights" (L28,L77); "9/10 diverge" (L292); win-5min
  37.0/15.1 (L250) vs Table III 33.0/13.5. Only bites a reviewer with repo access,
  but should be reconciled with the manuscript re-run (and with the T2-1' decision on
  13.1 vs 13.8).

- **T3-5. "roughly halves" overstates a 40% reduction.** §4 L948–949 "removes only
  about half ... to roughly 50 m" and L965 caption "roughly halves"; the actual
  reduction is 82.35→49.54 = **−39.8%**, and `fig_warmstart` itself annotates
  "−40%." "About half" is loose against a figure that says 40%. Either soften text to
  "cuts it by about 40%" or accept the mild tension; low priority but a picky reviewer
  will diff the caption against the figure.

- **T3-6. 30 s-lag "smoothed 13.4 m" vs "causal 16.5 m" may read as a contradiction.**
  §5.7 reports both the causal/zero-latency 30 s-lag DRMS (16.5, L1463) and the
  committed/smoothed 30 s-lag DRMS (13.4, L1479). Both are correct and sourced
  (`prog_lag30` warm=300: realtime 16.53 / smoothed 13.36), but the reader must track
  that "30 s lag" yields two numbers depending on causal-vs-committed. One clause
  ("the *committed* estimate at the trailing edge of the lag, distinct from the
  zero-latency causal estimate above") removes the ambiguity.

- **T3-7. Statistical strengthening still cheap.** v3's suggestion to report a
  paired effect size `d_z` (and optionally Wilcoxon signed-rank, which uses the DRMS
  magnitudes already in Table IV) alongside the sign test remains unaddressed. Not
  required, but it upgrades the significance claim at zero data cost.

---

## Numeric 3-way re-audit (manuscript ↔ CI result files ↔ state.md)

**Verified consistent this pass:**

| Quantity | Manuscript | Source file | ✓ |
|---|---|---|---|
| Realtime 30 s lag, cold/warm (t≥300) | 16.5 / 15.1 (§5.7 L1463–1464) | `prog_lag30`/`warm_lag30` warm=300: 16.53 / 15.14 | ✓ |
| 300 s-lag smoothed (t≥300) | 11.0 (§5.7 L1464, L1480) | `prog` warm=300: 11.01 | ✓ |
| 30 s-lag smoothed (t≥300) | 13.4 (§5.7 L1479) | `prog_lag30` warm=300 smoothed: 13.36 | ✓ |
| Warm-start transient (t≥60) | 82.4 / 49.5 (§4, §5.7 L1460–1461) | `prog_lag30`/`warm_lag30` warm=60: 82.35 / 49.54 | ✓ |
| fig_realtime legend (drms300) | 16.5/15.1/11.0/52 | recomputed at warm_s=300 from npz-backed txt | ✓ (now matches text) |
| Cross-impl full line GTSAM | 12.4 (§5.6) | `window_full` warm=600: 12.41 | ✓ |
| Cross-impl segment GTSAM/Julia | 17.5 / 21.4 (§5.6) | `window` warm=60: 17.47 / `julia_ref` 21.395 | ✓ |
| INS full line / segment | 318 / 317.5 (§5.6) | `window_full` warm=600 ins 317.50 | ✓ |
| Speed | 14 ms/epoch (§5.7, Abstract) | 83 s / 6000 = 13.8 ms | ✓ |
| MC DRMS / ANEES | 1.4/4.1/4.9; 1.57/1.72/59.8 | montecarlo_summary (per v3) | ✓ |

**Residual discrepancies (all traced to the v3 fix or to artifacts):**

| # | Quantity | Manuscript | Artifact | Tier |
|---|---|---|---|---|
| A | §5.6 full-line Julia | 13.1 (from Table IV re-run) | `julia_ref`/README/script/state.md = 13.8 (same run as the 21.4 segment) | **T2-1'** |
| B | c_t definition | A^Tβ (L102, Eq 5) vs A^Tβ+S_t (L450) | — | **T2-2'** |
| C | research/README breadth column | 30.7/13.1 (Table IV) | README 32.7/13.8 | T3-4 |
| D | "four" vs "three" estimators | Abstract/Contrib/L1294 "four"; caption/L1183 "three" | README "best of 4" | T3-1 |

---

## Grep pass summary

- **Em-dashes (`---`):** none. (v3 T3-1 cleared.)
- **"sole/single design variable":** none; only "sole source of horizontal position"
  (L457, legitimate). (v3 T2-2 cleared.)
- **`% TODO` / placeholder:** L134, L1620 remain (T3-3). No `[...]` / `XXX` in body
  (only template stubs L19/L46).
- **Section-letter refs:** L657→D ✓, L946→G ✓ (v3 T3-4 targets fixed); **L1352→"B
  and C" is newly wrong, should be "C and D"** (T3-2).
- **"four"/"three":** inconsistent across L75/L366/L1294 (four) vs L1183/L1232 (three)
  — T3-1.
- **13.8 in .tex:** none (all 13.1); 13.8 survives only in artifacts (T2-1').

---

## Bottom line for the authors

The revision did its main job: the two v3 Tier-1 levers — the fig↔text realtime
contradiction and the Julia three-value inconsistency — are gone from the reader's
view, verified by independent recompute, and **no new Tier-1 was introduced.** That
alone moves the paper off the desk-reject margin.

Three fixes are worth doing before a reviewer or a repo-access auditor sees it, in
order:

1. **T2-1' — un-splice §5.6.** Either report full-line 12.4 vs **13.8** (same run as
   the 21.4 segment) with a one-clause note that the current Huber window is 13.1
   (Table IV), or re-run GTSAM against the 13.1 config and update
   `julia_ref`/README/`state.md`/the script constant. Also confirm both windows use
   the same Huber setting before claiming "GTSAM the lower." This is the only place
   the v3 fix pushed the problem sideways rather than closing it.
2. **T2-2' — fix c_t.** Write Eq (3) as `z = h + c_t + S_t + η` and keep c_t = A^Tβ;
   delete the "decomposed into S_t" clause at L450. One line, closes both T3-3 and
   the new overload.
3. **T3-1/T3-2 — one count, one letter.** Make Abstract/Contribution/§5.4 agree on
   "three tabulated estimators" (MPF separate), and change L1352 "B and C" → "C and D."

After those, the standing Major-Revision items are the ones a reviewer will still
raise but that do not sink the paper: the single-run authors'-own NN baseline
(T2-3 — add a run spread), the missing no-Huber Table IV column (T2-4 — one rerun),
the 30-seed (not 100+) Monte-Carlo, and the pre-submission housekeeping (TODO cite,
co-author bio, stale research/README). The consistency study, the divergence-
avoidance result, the `:filtered` handoff, and now a numerically self-consistent
real-time/cross-impl narrative are the defensible core.
