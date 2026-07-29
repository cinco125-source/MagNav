# Red Team Report

**Target:** `paper/taes_fgo_magnav.tex` — "Factor-Graph Aeromagnetic Compensation for Airborne Magnetic-Anomaly Navigation," submitted to IEEE TAES.
**Modes applied:** Mode 1-Nav (2A/2B/2C/2D incl. factor-graph checklist), Mode 2 (common + N1–N12), Mode 4 (ablation honesty, sensor-error section), 수치 3-way 일관성 감사 (manuscript ↔ `paper/make_figures.py`+`plot_tracks.py` ↔ `research/README.md`), placeholder/label Grep, code-vs-manuscript audit against `src/fgo_online.jl`, `src/fgo.jl`, `src/fgo_sensor.jl`.

---

## Summary

논문 분야: 항법/추정 (MagNav, factor-graph fixed-lag smoothing, joint aeromagnetic compensation)

Accept probability: **~10% for IEEE TAES**
산출: TAES baseline 25% · Tier 1 이슈 4개 (2+ ⇒ −30%) · 실비행 데이터 실험 (+15%) · Theorem/Proof 완전 (+10%) **불충족** — Proposition 1에 논리 결함 · Monte Carlo 100+ (+10%) 없음 · NEES/NIS (+10%) 없음 · ±2σ plot (+5%) 없음 ⇒ 25 − 30 + 15 = **10%**.

Expected decision: **Reject** (운 좋으면 Major Revision). Tier 1의 4건 중 3건(Prop. 1 수리 결함, Huber 교란 비교, 발산 집계 불일치)은 리뷰어 1명만 발견해도 신뢰도 전체가 무너지는 유형이다. 반대로 4건 모두 원고 수정만으로 방어 가능하므로, 수정 후 재산정 시 Tier 1 = 0, +15% 비행데이터, +10% proof 완전 ⇒ ~60%대까지 회복 가능한 논문이다.

Positives (공격 아님, 기록용): 실비행 5개 라인 breadth, 재현 baseline, negative result(screening statistic 실패)의 정직한 보고, placeholder/undefined-ref/중복 label **0건** (Grep 확인), 모든 \cite 키가 references.bib에 존재.

---

## Tier 1: Desk-Reject Risk

### 1. [공격 코드: 수학적 결함 #1 + N3 / Mode 1-Nav 2B] Proposition 1의 "iff"는 거짓이고, 논문 자신의 Remark와 모순된다

**구체적 내용.** Proposition 1 (Sec. IV)의 두 번째 문장:

> "Consequently the joint state $(\delta\mathbf{p},\delta\boldsymbol{\beta})$ is observable **if and only if** $\mathrm{range}(\mathbf{G})\cap\mathrm{range}(\boldsymbol{\Psi})=\{\mathbf{0}\}$."

이 조건은 관측가능성의 **필요조건일 뿐 충분조건이 아니다.** 반례가 논문 안에 있다: 평탄한 맵에서 $\mathbf{G}\equiv\mathbf{0}$이면 $\mathrm{range}(\mathbf{G})=\{\mathbf{0}\}$이므로 교집합은 자명하게 $\{\mathbf{0}\}$ — Proposition 1은 "observable"을 선고한다. 그러나 Sec. IV의 Remark 자신이 "over a locally flat map $\mathbf{G}\approx\mathbf{0}$ and *no* compensation can be separated from position"이라고 쓴다. **Proposition과 Remark가 정면 충돌한다.** 원인은 증명의 이 문장이다:

> "A nonzero position difference with this property exists iff $\mathrm{range}(\mathbf{G})\cap\mathrm{range}(\boldsymbol{\Psi})\neq\{\mathbf{0}\}$"

거짓이다. $\delta\mathbf{p}-\delta\mathbf{p}'\in\ker(\mathbf{G})\neq\{\mathbf{0}\}$이면 $\mathbf{G}(\delta\mathbf{p}-\delta\mathbf{p}')=\mathbf{0}\in\mathrm{range}(\boldsymbol{\Psi})$가 항상 성립하므로, 교집합이 자명해도 그런 nonzero 차이가 존재한다. 올바른 동치는 증명 마지막 구절이 이미 알고 있다: "$[\,\mathbf{G}\;\boldsymbol{\Psi}\,]$ is column-rank-deficient" — 즉 **joint observability ⟺ $[\mathbf{G}\;\boldsymbol{\Psi}]$ full column rank ⟺ ($\ker\mathbf{G}=\{0\}$) ∧ ($\ker\boldsymbol{\Psi}=\{0\}$) ∧ (range 교집합 자명)**. 현재 Proposition·증명·"equivalently" 구절이 서로 다른 세 조건을 동치라고 주장한다.

**Corollary 1도 이 결함을 상속한다.** "the joint problem is observable **at any basis dimension**"은 $\ker\boldsymbol{\Psi}=\{0\}$을 요구하는데, 논문 자신의 기본 basis(Appendix B: 18항 + bias, 코드 기본값 `terms=[:permanent,:induced,:eddy,:bias]`, `src/fgo_online.jl` L87)는 $u^2+v^2+w^2=1$ 때문에 유도자화 대각 3열의 합 $B_t(u^2+v^2+w^2)=B_t$가 bias 열과 공선 — **TL 회귀의 고전적 rank 결핍** [Leliak 1961 이후 주지]. 즉 논문의 기본 설정에서 $\boldsymbol{\Psi}$는 구조적으로 column-rank-deficient이고, Proposition 1의 엄밀한 의미로는 $\delta\boldsymbol{\beta}$가 관측 불가능하다. 또한 "generically transverse"는 정의·증명 없이 선언되며(Mode 1 체크리스트의 전형적 hand-waving), 차원 조건($\dim$ range G + $\dim$ range Ψ ≤ 윈도 길이 $N$)조차 명시하지 않는다. 마지막으로, 윈도 모델 $\mathbf{z}=\mathbf{G}\delta\mathbf{p}+\boldsymbol{\Psi}\delta\boldsymbol{\beta}+\boldsymbol{\eta}$는 윈도 내내 $\delta\mathbf{p}$, $\delta\boldsymbol{\beta}$를 **상수로 고정**한다 — 실제 추정기는 식 (2)의 $\boldsymbol{\Phi}_t$ 동역학과 식 (7)의 random-walk $\boldsymbol{\beta}_t$를 쓰므로, 이것은 18+m 상태 시스템의 observability가 아니라 정적 2-블록 회귀의 식별가능성이다. 유일한 이론 기여가 이 상태면 TAES 리뷰어는 멈춘다.

**방어 (Scope + rewrite, 실험 불필요):**
1. Proposition 1을 올바르게 재서술: "unobservable subspace of $(\delta\mathbf{p},\delta\boldsymbol{\beta})$ = $\ker[\mathbf{G}\;\boldsymbol{\Psi}]$; 특히 position이 compensation과 confound되는 것은 $\mathbf{G}\delta\mathbf{p}\in\mathrm{range}(\boldsymbol{\Psi})$, $\delta\mathbf{p}\notin\ker\mathbf{G}$인 경우" — 첫 문장은 살리고 "Consequently" 문장만 3-조건 형태로 교체.
2. Corollary를 "position–compensation **separability** at any basis dimension"으로 완화하고, $\boldsymbol{\Psi}$ rank 결핍(TL 공선성)은 $\delta\boldsymbol{\beta}$ 내부의 gauge freedom일 뿐 $c_t=\mathbf{A}_t^\top\boldsymbol{\beta}_t$와 position에는 영향 없음을 Remark로 명시.
3. "generically transverse"에 최소한의 조건 부여: attitude 시계열이 map-gradient 시계열과 선형종속이 아닐 것 + $N \ge \dim$ 조건.
4. 정적 윈도 모델은 "per-window, frozen-state surrogate"라고 명시적 scope 한정.

**공수:** 1–2일 (수학 재작성 + 관련 문장 5곳 정합화; 코드/실험 변경 없음).

---

### 2. [공격 코드: 비교 불공정 #4] Breadth 비교에서 FGO만 Huber를 쓰면서 "only the estimator structure differs"라고 주장

**구체적 내용.** Sec. VI-A의 프로토콜 선언:

> "For every comparison the EKF and FGO estimators use identical dynamics, map interpolation, and noise settings, so that only the estimator *structure* differs."

그러나 Table III caption은 "causal online-TL EKF vs. 5 min-window smoother **(Huber)**"이다 — **FGO에는 robust kernel을 주고 EKF에는 주지 않았다.** 이것이 사소하지 않음은 논문의 provenance가 스스로 증명한다: `research/README.md` §2에서 같은 cold-start Mag 4 케이스에 FGO-online 60.1 m vs **FGO-online+Huber 22.6 m** — Huber 하나가 2.7배를 좌우한다. Robust/Huber EKF는 표준 기법이므로(예: Huberized Kalman update), Table III의 EKF 발산 3–4건 중 몇 건이 "causal이라서"가 아니라 "robust kernel이 없어서"인지 현재 실험으로는 분리 불가능하다. 논문의 핵심 주장("remains bounded where the causal filter diverges", Abstract)이 교란 변수 위에 서 있다.

추가 교란: $L_w=5$ min은 Fig. 10(fig:winlen)의 sweep으로 **line 1007.06에서 선택**되었는데, 1007.06은 Table II의 primary line이자 Table III breadth 집합의 평가 라인이다 — 하이퍼파라미터를 평가 데이터에서 튜닝한 뒤 같은 데이터에 보고(선택 편향).

**방어 (Experiment, 스크립트 재실행):**
1. Table III에 두 열 추가: **FGO-win(no Huber)** 와 (가능하면) **Huberized EKF-online**. `research/fgo_breadth.jl`에 `robust=:none` 런은 kwarg 하나로 즉시 가능. FGO(no Huber)가 여전히 bounded면 주장은 오히려 강해진다.
2. 최소 방어(코드 추가 없이): "identical ... noise settings" 문장을 "identical dynamics, map interpolation, and noise settings; the smoother additionally uses the Huber kernel of (10), whose contribution is isolated in Table II (37.0→32.6 m)"로 교체하고, Table III에 no-Huber FGO 열만 추가.
3. $L_w$ 선택: leave-one-line-out으로 재확인하거나 "selected on 1007.06 and held fixed for all other lines"를 명시 (후자는 문장 1개).

**공수:** CI 재실행 포함 1–2일 (no-Huber 열), Huber-EKF까지 구현 시 +2–3일.

---

### 3. [공격 코드: Table 불일치 #7 / 3-way 감사] 발산 케이스 집계가 본문·Table III·provenance 세 곳에서 서로 다르다

**구체적 내용.** 3-way 감사에서 가장 심각한 발견. 원문(Sec. VI-D):

> "On the three lines where the causal EKF diverges (1003.02, 1003.08, and 1007.02 on Mag~4), the divergence is to $1.7$–$42\times10^{3}$\,m"

- (a) **Table III에는 "div."가 네 줄이다** (1003.02, 1003.08, 1006.08, 1007.02의 Mag 4). 본문은 두 곳에서 "three lines"라고 말한다(VI-D와 VI-D 첫 단락 "diverges from the cold start to tens of kilometers on three lines"). 자기 표와 불일치.
- (b) **"$1.7$–$42\times10^{3}$ m" 범위가 provenance와 안 맞는다.** `research/README.md` §2b의 실측 발산치는 41626 m (1003.02), 35482 m (1007.02), 17179 m (**1006.08**)이고 — 1.7×10³ m에 해당하는 수치는 존재하지 않는다(17179의 오독으로 추정: 17.2×10³). 게다가 수치가 있는 세 라인은 (1003.02, 1006.08, 1007.02)이지 본문의 (1003.02, **1003.08**, 1007.02)가 아니다.
- (c) **1003.08 Mag 4은 발산이 측정된 게 아니라 런이 죽었다.** README §2b: 1003.08 Mag 4 EKF는 "diverged ✗" (수치 없음), 각주 "**one EKF run errored off-map**". Table III caption은 " 'div.' denotes divergence beyond 10 km"라고 정의하므로, crash한 런을 측정된 >10 km 발산처럼 제시한 것 — 데이터 표현 무결성 문제로 읽힐 수 있다.
- (d) **"8 of 9" 집계의 근거가 원고에 없다.** Table III는 10행이다. div.를 FGO 승리로 세면 9/10이다. 8/9는 README의 "(one EKF run errored off-map)" — 즉 1003.08 M4 제외 — 에서만 설명되는데, 원고는 그 제외를 어디에도 밝히지 않는다. Abstract의 "improves ... on 8 of 9 cold-start cases"를 리뷰어가 표에서 재구성하지 못한다.

**방어 (원고 수정만):** "three"→"four" (또는 1006.08을 별도 언급), 범위를 "17–42×10³ m"로 수정, 1003.08 M4를 "err." 등 별도 기호로 구분하고 caption에 "the EKF run on 1003.08/Mag 4 exited the map region and is excluded from the 8-of-9 count" 1문장 추가. 전부 텍스트 수정.

**공수:** 반나절.

---

### 4. [공격 코드: 수학적 결함 #1(미정의 참조) / 투고 직전 체크] 핵심 baseline 인용 `hager2026`가 placeholder 서지정보다

**구체적 내용.** `paper/references.bib` L61–69:

```
@article{hager2026,
  author  = {Hager and others},
  ...
  note    = {Citation metadata to be finalized against the published version.}
```

이 논문의 **비교 대상 그 자체**(Abstract, Contribution 3, Sec. V 전체, Table II의 published 37/14/58/15가 전부 여기서 나온다)가 저자명 "Hager and others" + "to be finalized" 상태다. arXiv 2603.08265라는 eprint만 있고 저자·제목의 검증이 불가능하면, 리뷰어는 Table II의 "published" 행 수치(37/14, 58/15)를 확인할 수 없다. 투고 시 desk editor가 잡는 전형적 사유이며, 과거 프로젝트에서 실제 R1 리젝 원인이었던 placeholder 잔존과 같은 클래스다 (tex 본문에는 placeholder 0건 — bib에만 남았다).

**방어:** 실제 서지사항으로 교체(저자 전원, 정확한 제목, arXiv/게재 상태). published 수치 37/14/58/15가 해당 논문의 어느 표/그림에서 왔는지 Sec. V 또는 Table II caption에 명시.

**공수:** 1시간.

---

## Tier 2: Major Revision Risk

### 5. [공격 코드: N1 + N2 + N4] 일관성 분석 zero — NEES도, NIS도, ±2σ plot도, Monte Carlo도 없다

진리값(SGL reference truth)이 있는 데이터에서 추정기의 covariance를 한 번도 검증하지 않았다. Fig. 6(fig:poserr)은 오차 궤적만 있고 $\pm2\sigma$ 경계가 없으며(`plot_tracks.py fig_poserr`에 covariance 입력 자체가 없음), NEES/NIS는 원고 전체에 부재. 이것은 단순 관행 문제가 아니라 **공격 6과 결합해 치명적이 된다**: 슬라이딩 윈도가 정보를 이중계상하면(아래) $P$는 과소(overconfident)해지는데, 그것을 드러낼 유일한 진단(NEES)이 없다. 또한 Sec. VI-D의 "the difference is within run-to-run variability" (1006.08 M5, 117.5 vs 122.0)는 **run-to-run 변동성을 한 번도 측정하지 않은 논문**(single-run, Sec. VII-D 자인)이 쓸 수 없는 문장이다. Limitations에서 MC 부재를 인정했지만, TAES 리뷰어는 인정 여부와 무관하게 요구한다 — 특히 8/9 같은 count 기반 주장에는 최소 sign test (8/9 승리면 $p\approx0.02$)라도 있어야 한다.

**방어:** (i) 1003.02와 1007.06에서 NEES/NIS + χ² 95% 구간 plot 추가 (truth 있음, 계산은 후처리만으로 가능 — `FILTres.P`가 이미 반환됨); (ii) fig_poserr에 ±2σ 밴드 추가; (iii) 초기조건 20–50 run MC 또는 최소한 paired sign test. **공수:** NEES/2σ 2–3일, MC는 CI 시간 포함 1주.

### 6. [공격 코드: Factor Graph 체크 / 수학적 결함] 윈도 간 정보 이중계상 — 원고는 정반대를 주장한다

Sec. III-B:

> "Carrying the full covariance---not just the mean---prevents the double-counting of information that a naive re-linearization would incur"

코드는 반대를 한다. `src/fgo_online.jl` L287–289: 다음 윈도의 prior는 commit 경계의 **smoothed** 상태 `x0_pr = res.x[:,lc1]`, `P0_c = res.P[:,:,lc1]` — 이 smoothed 추정치는 overlap 구간 $[s{+}L_w{-}L_o, s{+}L_w)$의 측정치를 **이미 포함**한다(RTS backward pass). 그런데 윈도 $k{+}1$은 같은 overlap 측정치를 다시 처리한다(L274–279: 새 윈도가 `i0+stride`에서 시작해 overlap 구간을 재포함). 즉 overlap의 $L_o$개 측정이 prior에 한 번, factor로 한 번 — **두 번 계상**된다. 통계적으로 올바른 fixed-lag 처리는 (i) filtered(forward-only) 상태를 carry하거나 (ii) overlap 측정을 다음 윈도에서 제외하거나 (iii) marginalization으로 prior를 구성하는 것이다. DRMS에는 영향이 작을 수 있으나 $P$는 체계적으로 과소해지고, "prevents the double-counting" 문장은 코드 기준으로 거짓이다. (부수: `fgo_online_window` docstring L245 "the overlap region is used as warm-up and discarded"는 자기 코드와도, 원고와도 다르다 — trailing look-ahead가 맞다.)

**방어:** 코드 수정이 정공법(carry를 forward-filtered 상태로 교체 — `fgo_rts_pass`가 `x_upd/P_upd`를 이미 계산하므로 반환만 추가하면 됨) + 재실행. 최소 방어는 문장을 "the committed estimates are exact fixed-lag smoothed estimates; the carried prior re-uses the overlap measurements, a standard approximation whose effect we bound by..."로 정직하게 교체하고 NEES(공격 5)로 영향 정량화. **공수:** 코드 수정+재실행 3–4일, 문장 수정만이면 반나절 (단 공격 5와 묶어 처리 권장).

### 7. [공격 코드: 수학적 결함 #1 / Mode 1-Nav 2A] 측정 Jacobian: 원고의 3-D $\mathbf{g}_t$ vs 코드의 2-D (수직 기울기 폐기)

식 (4)와 식 (9)는 $\mathbf{g}_t=\partial h/\partial\mathbf{p}$ — $\delta\mathbf{p}\in\mathbb{R}^3$에 대한 **3성분** 기울기를 선언한다. 그러나 cold-start 결과 전부를 만든 `fgo_online`의 Jacobian 조립(L171)은

```julia
H_bar[:,t] = [Hll[1:2]; zeros(eltype(P0),nx-3-nx_TL); A[t,:]; 1]
```

— `get_H`가 반환한 3-D 기울기(`src/model_functions.jl` L525–534, lat/lon/alt) 중 **처음 2성분만** 쓰고 고도 성분 $\partial h/\partial\mathrm{alt}$는 0으로 둔다(인덱스 3이 zeros 블록에 흡수). 반면 stinger 결과(Table I)를 만든 `fgo.jl`은 `get_H` 전체를 쓴다(L163). 즉 두 추정기의 측정 모델이 다르고, 둘 다 식 (9) 그대로는 아니다. upward-continued 맵에서 수직 기울기는 작지 않다. 부수적으로 상태 순서도 불일치: 식 (1)은 $S$를 $\mathbf{x}\in\mathbb{R}^{18}$의 18번째 성분으로 두지만 코드의 joint 상태는 $[\,\text{Pinson 17};\ \boldsymbol{\beta}\ (18{:}17{+}n_\beta);\ S\ (\text{마지막})\,]$이다 — 식 (9)의 블록 순서 $[\mathbf{g}^\top, \mathbf{0}, \mathbf{A}^\top, 1]$은 코드와 맞지만 식 (1)·Nomenclature와는 어긋난다.

**방어:** 식 (9)에 각주 또는 문장: "the vertical gradient is excluded from the online estimator (baro-damped vertical channel; the map is evaluated at the baro altitude)" — 실제 이유를 확인해 명시. 상태 순서는 식 (1) 아래 한 문장으로 정리. **공수:** 반나절 (서술 정합화) 또는 코드 통일 후 재실행 2일.

### 8. [공격 코드: 비교 불공정 #4 + Novelty] NN baseline 비교는 n=1 라인이고, "faithful reproduction"과 "inside the band"는 Mag 5에서 수치로 반박된다

- **"inside the band published for it"** (Sec. V): 재현 결과 40.0/**17.5** m. published 값은 TL+NN 37/**14**, TL-only 58/**15** (Table II). Mag 4의 40.0은 [37, 58] 안이지만, **Mag 5의 17.5 m는 published 어떤 값(14, 15)보다도 나쁘다 — band 밖이다.** 문장이 수치로 거짓.
- 재현 필터는 permanent 그룹 + 8-unit(41 weights) NN이고, README §5b가 자인하듯 "we do not have their released architecture / natural-gradient stabilization". 이것은 reproduction이 아니라 **동일 계열의 재구현**이며, published보다 3–3.5 m 나쁘다. 그런데 FGO의 마진(40.0→32.6)은 그 재현 격차(40.0 vs 37)와 같은 크기다 — 제대로 튜닝된 baseline 상대로는 마진이 사라질 수 있다.
- Abstract "matches or beats a reproduced online EKF+TL+NN **and the values published for it**": published 대비 Mag 5는 14.2 vs 14 — **0.2 m 더 나쁘다.** single-run·오차막대 없음 상태에서 "matches or beats"는 과잉.
- NN baseline은 **1007.06 한 라인**에서만 실행되었다. Table III breadth의 비교 상대는 TL-only EKF다. 헤드라인 주장("reaches the same accuracy without a neural network")의 실증 기반이 n=1.

**방어:** (i) "inside the band"→"inside the published Mag 4 band; on Mag 5 it is 3.5 m above the published filter, consistent with our lack of their stabilization details"; (ii) "reproduced"→"reimplemented (same filter family, architecture details unavailable)"; (iii) abstract를 "matches (within 0.2 m) or beats"로 정밀화; (iv) 가능하면 `research/paper_impl.jl`을 breadth 5개 라인에 확장 — kwarg 수준 작업. **공수:** 문장 수정 반나절; NN baseline breadth 확장 2–3일.

### 9. [공격 코드: Mode 4 ablation honesty + 시뮬레이션 부족 #3] Sensor-error 절: 자기 표와 모순되는 서술 3건 + model-matched 주입 회수

Sec. VI-E / Table IV (synthetic-only 공격은 이 절에만 적용):
- **"drives the error down monotonically to 4.8 m" — 거짓.** Table IV: 19.1 → **19.2** (+ dead-zone weight에서 0.1 m 증가). 단조가 아니다.
- **"The largest gains come from the hard-iron bias and the linear drift" — 자기 표와 모순.** 누적 표의 최대 단일 감소는 heading harmonics다(36.6→19.1, −17.5 m; hard-iron −10.8, drift −3.6). 저자의 의도(파라미터 회수 품질)와 DRMS 기여가 뒤섞였다. 누적(단일 순서) ablation에서는 순서가 기여를 교란하므로 marginal 귀속 자체가 불가 — README는 스크립트를 "factorial sensor-error ablation"이라 부르는데 **factorial 결과는 원고 어디에도 없다.**
- **Dead-zone 항은 (i) 그래프 변수가 아니고 (ii) 유일한 실험에서 inert다.** Sec. III-F 리드 문장은 "four ... terms, **each appearing additively in the residual**"인데 4번 항목 dead-zone weight는 residual에 additively 등장하지 않는 heteroscedastic **가중치**다(자기 문장 내 모순). Contribution 4("graph variables that a position-only estimator cannot represent")에도 안 맞는다 — 측정 de-weighting은 position-only 추정기도 할 수 있다. 게다가 README §3: "inert here (level flight never enters a dead zone)" — 논문이 제시한 유일한 검증에서 이 factor는 아무것도 안 했다(+0.1 m). Insurance를 contribution으로 부풀린 사례. 그리고 원고는 $R_t^{-1}=R^{-1}\omega(z_t)$ — **측정값 $z_t$의 함수**라고 쓰지만 코드는 $\omega$가 sensor–field angle $\theta_t$의 함수다(`src/fgo_sensor.jl`: `R/|sin 2θ|²`, `dz_floor=0.05`). 표기 오류.
- **Synthetic-only 표준 공격:** 주입된 오차(정확한 Fourier heading, 정확한 상수 bias, 정확한 선형 drift)가 추정기의 모델과 **동일 함수형**이다 — "주입한 것을 복원했을 뿐"이며 모델 불일치 0의 최상 시나리오. Limitations가 "simulation only"는 인정하지만 model-matched라는 더 날카로운 사실은 인정하지 않는다.

**방어:** "monotonically" 삭제, largest-gains 문장을 "under this enabling order"로 한정 + factorial 표(스크립트 이미 존재)를 부록에, dead-zone을 변수 목록에서 분리해 "heteroscedastic weighting (not an estimated variable)"로 강등, $\omega(z_t)$→$\omega(\theta_t)$, Limitations에 "the injected errors share the estimator's functional form; structured model mismatch is untested" 1문장. **공수:** 1일 (factorial 표 포함 2일).

### 10. [공격 코드: 시뮬레이션 부족 #3 / C2] 윈도 길이 "U-curve"는 데이터 3점이고, provenance에는 반례 라인이 있다

Fig. 10(fig:winlen)의 x축은 {2, 5, 87} min — **3점**이며 그중 1점은 퇴화된 full-line static이다. "The curve is U-shaped rather than monotone"과 "sweet spot"은 3점으로 성립하지 않는다(중간점이 최저인 3점 집합은 전부 'U'다). 10/20/40 min이 없다. 더 아픈 것: `research/README.md` §2는 **1003.02 Mag 4에서 batch+Huber 22.6 m vs Table III의 5-min window(Huber) 42.6 m** — 다른 라인에서는 static batch가 윈도를 ~2× 이긴다는 provenance가 존재하는데 원고는 1007.06의 sweep만 보여준다. 윈도의 이점이 라인 의존적일 가능성을 원고가 은폐한 모양새가 된다.

**방어:** $L_w\in\{2,5,10,20,40,87\}$ sweep을 1007.06과 1003.02 두 라인에서 실행(간단한 kwarg sweep, CI 1회) — U-curve가 진짜면 주장이 강해지고, 1003.02 결과는 "window advantage is line-dependent; the batch limit can win when interference is quasi-static"으로 정직하게 수용. **공수:** 2일.

### 11. [공격 코드: N6] 제안 기법의 노이즈/커널 파라미터가 원고에서 재현 불가

Baseline에는 수치가 있다(FOGM $(\sigma,\tau)=(3\,\mathrm{nT},180\,\mathrm{s})$, $R=12^2\,\mathrm{nT}^2$, Sec. V). 그런데 **제안 기법**의 $\mathbf{Q}^{\beta}$(원고가 Sec. VII-C에서 "main tuning burden"이라 부르는 바로 그 것), $\mathbf{P}_0$의 diffuse β 블록 크기, $R$, FOGM $(\sigma,\tau)$, Huber $\delta$(코드 기본 1.345, `robust_c`, `src/fgo.jl` L134) 중 **단 하나도 수치로 주어지지 않는다.** "identical noise settings" 주장(공격 2)과 결합하면, 리뷰어는 어떤 설정이 공유되었는지조차 확인할 수 없다. IRLS 세부(가중치가 2번째 iteration부터 적용, $[10^{-6},1]$ clamp — `src/fgo_online.jl` L209–213)도 식 (11)의 서술("each Gauss–Newton iteration reweights")과 다르다.

**방어:** 파라미터 표 1개(Table: $P_0$ 블록, $Q_d$, $\mathbf{Q}^\beta$, $R$, FOGM, $\delta=1.345$, $L_w$, $L_o$, n_iter, tol) 추가 — 값은 전부 코드/CI에 있다. **공수:** 반나절.

### 12. [공격 코드: N5 + N7 + N11] 초기오차 민감도·맵 오차 민감도·실측 timing 부재

(i) "Cold start"는 TL 계수만 0으로 초기화 — **초기 위치/자세 오차** 민감도는 미시험 (100 m vs 1 km 초기 오차에서 윈도 스무더의 수렴 반경은?). 특히 Prop. 1의 collapse 논의와 결합하면 큰 초기 오차에서 wrong-basin 수렴이 우려된다. (ii) 맵 품질 의존성은 FOGM state가 흡수한다는 정성 서술뿐, 50 nT급 맵 오차 주입 실험 없음 (N7 표준 공격). (iii) "runs comfortably faster than real time" (Sec. VI-F)에 숫자가 없다 — README에는 batch 런타임(EKF 23 s / FGO-RTS 18 s, 63-min 라인)이 있는데 원고는 그마저 안 실었고, **윈도 스무더**의 per-commit 시간은 어디에도 없다. $O(L_w)$ 표기도 블록 차원 $(n+m)^3$ 인자를 숨긴다.

**방어:** (i) 초기오차 sweep 1개 라인 (kwarg 수준, 1–2일); (ii) 맵 오차 주입 sweep (시뮬레이션, 1–2일); (iii) 윈도 per-commit wall-clock을 표로 (반나절 — CI 로그에서 추출 가능). Remark 템플릿("Q was derived from...", "Fig. X shows degradation under [10,50,100] nT") 활용.

### 13. [공격 코드: Factor Graph 체크 / 수치 감사] Algorithm 1은 sparse QR을 명시하지만 cold-start 결과는 전부 RTS 경로로 생성되었다

Algorithm 1 line 7: "solve normal equations by sparse QR (square-root SAM)". 그러나 `fgo_online`은 **`fgo_rts_pass`만 호출한다** (`src/fgo_online.jl` L216–217) — GN/QR 경로(`fgo_gn_step`)는 TL 변수가 없는 `fgo`에만 존재한다(`src/fgo.jl` L186). 즉 Table II·III의 모든 cold-start/breadth 수치는 Algorithm 1이 명시한 solver로 계산되지 않았다. 원고의 방어선(Sec. III-E "we verify the two agree to sub-metre level")도 구멍이 있다: 그 검증은 (i) **TL 변수 없는** stinger 케이스, (ii) 10-min segment에서만 이루어졌고(Table I), (iii) 그 일치 폭이 Table I 기준 19.3−18.4=0.9 m로 간신히 sub-metre인데 **README §2는 같은 수치를 19.4로 기록**한다(19.4−18.4=1.0 m — sub-metre 주장 자체가 0.1 m 차이로 갈린다; 3-way 감사 불일치 #1). joint(TL 포함) 문제에서 두 solver의 일치는 한 번도 검증되지 않았다.

**방어:** Algorithm 1 line 7을 "solve by iterated RTS pass (equivalent to sparse-QR square-root SAM on the chain, Sec. III-E)"로 교체하거나, `fgo_online`에 `solver=:gn` 경로를 추가해 1개 윈도에서 일치 확인. 19.3/19.4는 원본 CSV 확인 후 한쪽으로 통일. **공수:** 서술 수정 반나절; GN 경로 추가 검증 2일.

---

## Tier 3: Polish

1. **[3-way 감사 잔여 목록]** (a) Table I의 "FGO (GN/QR) 19.3" vs README §2 "19.4" — 0.1 m 불일치 (공격 13에 병합했으나 표 자체도 수정 필요). (b) Table I의 EKF 10-min 47.9와 GN/QR+Huber 18.4는 README에 없음 — provenance 문서에 추가하거나 CI CSV 참조 명시. (c) 그 외 전 수치 일치 확인 완료: Table II ↔ `make_figures.py fig_coldstart`(m4=[123.7,37.0,32.6,40.0,37.0], m5=[68.1,15.1,14.2,17.5,14.0]) ↔ README §5c ✓; Table III ↔ `fig_breadth` rows ↔ README §2b ✓ (발산 표기 제외, 공격 3); Table IV ↔ README §3 ✓; fig_winlen ↔ Table II ✓; 51% (28.3→14.0), 41 NN weights (3×8+8+8+1), 8 nT RMS (√((10²+6²)/2)=8.2), "28%" (per-case 감소 평균 27.8%) 재계산 일치 ✓ — 단 "28%"는 산출 방식(ratio-of-means면 17%)을 명시할 것.
2. **식 (7)에 β prior 항이 없다.** Sec. III-A 항목 1)은 "joint prior on $\boldsymbol{\chi}_1$"을 선언하고 코드도 $P_0$로 TL 블록을 anchor하는데(x0_TL, `fgo_online` docstring L28–30), 식 (7)의 prior 항은 $\lVert\mathbf{x}_1-\mathbf{x}_0\rVert^2_{\mathbf{P}_0^{-1}}$ — $\boldsymbol{\beta}_1$ prior가 빠져 있다. $\boldsymbol{\chi}$로 교체.
3. **Carried prior의 1-step 전파 누락.** 다음 윈도의 prior는 epoch $s{+}L_w{-}L_o{-}1$의 상태인데 새 윈도 첫 epoch은 $s{+}L_w{-}L_o$ — $\boldsymbol{\Phi}$ 전파 없이 한 epoch 어긋난 채 적용된다(`fgo_online_window` L287–289 vs L274). $dt=0.1$ s라 실질 영향은 미미하나 모델상 오류; Algorithm 1과 코드에 주석 한 줄.
4. **표기 불일치:** Sec. III-F heading 계수 $(a_k,d_k)$ vs 코드 $(a_k,b_k)$; 식 (6)의 induced 표기 "$B_tu^2,\dots,B_tw^2$"는 대각 3항처럼 읽힘(교차항 포함 6항은 Appendix B에서야 명시) — 식 (6)에 $uv,uw,vw$ 포함을 암시하는 표기로.
5. **DRMS 정의 vs 그림:** Sec. VI-A는 "after a 10 min warm-up"을 전 결과의 metric으로 선언하는데, `plot_tracks.py` fig_poserr/fig_cdf의 범례 DRMS는 CSV 전 구간 $\sqrt{\mathrm{mean}(e^2)}$다. track_data.csv가 warm-up을 포함하면 범례 수치가 Table I과 어긋난다 — CSV 생성부(`research/fgo_tracks.jl`) 기준 확인 후 caption에 구간 명시. Table I(batch, README §2)에 warm-up 적용 여부도 README와 원고가 서로 침묵/단언으로 갈린다.
6. **"convergence in one or two Gauss–Newton iterations"** (Sec. III-C) — 근거 수치 없음; per-window iteration count 히스토그램 한 줄이면 해결.
7. `hager2026`가 게재 전 preprint라면 "strongest **published** cold-start result" (Sec. I P5)의 "published"를 "reported"로.
8. Grep 결과 기록: tex 내 placeholder(`[...]`, TODO, ??) 0건, 중복 label 0건, 미정의 \ref 0건, bib 누락 cite 0건 — 유지할 것.

---

## 방어 전략 요약

(rebuttal 재사용을 위해 우선순위·의존관계 순)

| # | 공격 | 방어 조치 | 종류 | 공수 |
|---|---|---|---|---|
| 1 | Prop. 1 iff 결함 + Cor. 1 | $[\mathbf{G}\;\boldsymbol{\Psi}]$ full-column-rank 조건으로 재서술; Corollary를 separability로 완화; TL gauge-freedom Remark; frozen-state surrogate scope 명시 | Rewrite | 1–2일 |
| 2 | Huber 교란 비교 | Table III에 FGO(no-Huber) 열 (+가능시 Huber-EKF); "only structure differs" 문장 교체; $L_w$ 선택 절차 명시 | Experiment | 1–3일 |
| 3 | 발산 집계 불일치 | three→four, 1.7→17×10³, 1003.08 M4를 "err."로 분리, 8/9 산정 기준 명시 | Rewrite | 0.5일 |
| 4 | hager2026 placeholder | 서지 확정 + published 수치 출처 명시 | Rewrite | 1시간 |
| 5 | NEES/±2σ/MC 부재 | NEES+NIS plot 2개 라인, ±2σ 밴드, 초기조건 MC 또는 sign test | Experiment | 3일–1주 |
| 6 | Overlap 이중계상 | forward-filtered carry로 코드 수정+재실행 (정공법) 또는 정직한 근사 서술+NEES 정량화 | Code+Exp | 0.5–4일 |
| 7 | 2-D vs 3-D Jacobian | 수직 채널 제외 사유 명시 or 코드 통일; 상태순서 각주 | Rewrite | 0.5일 |
| 8 | NN baseline n=1·band 밖 | Mag 5 문장 수정; "reimplemented"로 표현 교정; 가능시 NN baseline을 5개 라인으로 확장 | Rewrite+Exp | 0.5–3일 |
| 9 | Ablation 서술 모순 | monotonic 삭제, 순서 한정, factorial 표 부록, dead-zone 강등, $\omega(\theta_t)$ 수정, model-matched 한계 1문장 | Rewrite | 1–2일 |
| 10 | U-curve 3점 | $L_w$ sweep 확장(2개 라인 × 6점), 1003.02 반례 수용 | Experiment | 2일 |
| 11 | 파라미터 미보고 | 파라미터 표 1개 (δ=1.345 포함) | Rewrite | 0.5일 |
| 12 | 초기오차/맵오차/timing | 초기오차 sweep, 맵오차 주입, per-commit wall-clock 표 | Experiment | 3–4일 |
| 13 | Algorithm 1 solver 불일치 | line 7 재서술 or joint-GN 검증; 19.3/19.4 통일 | Rewrite | 0.5–2일 |

**총 공수 추정:** Rewrite-only 경로(공격 2·5·6·10·12의 실험 없이) ≈ 5–6일로 Tier 1 전부 + Tier 2 대부분의 문장 결함은 소거 가능하나, 공격 2(no-Huber 열)와 5(NEES/±2σ)는 실험 없이는 리뷰어를 설득할 수 없다 — **최소 실험 패키지(no-Huber breadth 열 + NEES/2σ + $L_w$ sweep 확장) 포함 2–3주**를 권장. 이 패키지 적용 시 Tier 1 = 0, NEES/±2σ 가점 회복으로 재산정 Accept probability ≈ 60% (Major Revision 후 Accept 경로).
