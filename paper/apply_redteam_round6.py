#!/usr/bin/env python3
"""Round-6 red-team fixes, all verified against the data or the source.

Confirmed defects addressed:
  Pinson sign convention (code verified: F[4:6,7:9] = +[f x], F[7:9,15:17] = -Cnb)
  text 46.7/17.8 vs tables 46.3/17.6
  "two cases" beating converged -> three
  floor stated as 1e-16 times nominal when 1e-31 rad^2 nominal implies 1e+11
  "one and a half to two orders" -> the measured factor 24 to 82
  ablation prose quoting only the least sensitive column
  causal six-of-eight asserted without significance; smoothed count without effect size
  metric decline not separated from its dimension-counting null
  mu not identified as the profiled Fisher information
  Proposition 1 silent on the velocity and disturbance nuisance directions
  units of B_k never stated
  contribution 3 says [19] reproduced while V-B says it is not
"""
import io

TEX = "taes_incremental.tex"
src = io.open(TEX, encoding="utf-8").read()

PAIRS = [
    # ---- Pinson sign + convention -------------------------------------
    ("Pinson model in a local north-east-down frame~\\cite{titterton2004,groves2013}. With\n"
     "position, velocity, and attitude errors $(\\delta\\mathbf{p},\\delta\\mathbf{v},\n"
     "\\boldsymbol{\\psi})$ and sensor biases $(\\mathbf{b}_a,\\mathbf{b}_g)$, the continuous\n"
     "error dynamics are",
     "Pinson model in a local north-east-down frame~\\cite{titterton2004,groves2013}. The\n"
     "attitude error is defined by\n"
     "$\\hat{\\mathbf{C}}^n_b=(\\mathbf{I}-[\\boldsymbol{\\psi}\\times])\\mathbf{C}^n_b$,\n"
     "the convention of~\\cite{groves2013}. With\n"
     "position, velocity, and attitude errors $(\\delta\\mathbf{p},\\delta\\mathbf{v},\n"
     "\\boldsymbol{\\psi})$ and sensor biases $(\\mathbf{b}_a,\\mathbf{b}_g)$, the continuous\n"
     "error dynamics are"),
    ("+\\mathbf{F}_{vv}\\,\\delta\\mathbf{v}\n-[\\mathbf{f}^n\\times]\\,\\boldsymbol{\\psi}+\\mathbf{C}^n_b\\mathbf{b}_a,",
     "+\\mathbf{F}_{vv}\\,\\delta\\mathbf{v}\n+[\\mathbf{f}^n\\times]\\,\\boldsymbol{\\psi}+\\mathbf{C}^n_b\\mathbf{b}_a,"),
    # ---- text/table mismatch -------------------------------------------
    ("online-TL EKF (\\num{46.7}/\\SI{17.8}{\\meter})",
     "online-TL EKF (\\num{46.3}/\\SI{17.6}{\\meter})"),
    # ---- three, not two -------------------------------------------------
    ("Mag~5, while the causal transient is essentially unchanged. The two cases of\n"
     "Table~\\ref{tab:transient} that beat their converged values do so by leaning\n"
     "on the tight anchor, and the advantage disappears with it.",
     "Mag~5, while the causal transient is essentially unchanged. The three cases of\n"
     "Table~\\ref{tab:transient} whose transient DRMS is below their converged value\n"
     "do so by leaning on the tight anchor, and the advantage disappears with it."),
    # ---- floor units ----------------------------------------------------
    ("The largest single effect is the process-noise floor: raising it from\n"
     "$10^{-16}$ to $10^{-12}$ times the nominal position channel, i.e.\\ from\n"
     "$10^{-20}\\mathbf{I}$ to $10^{-16}\\mathbf{I}$, costs \\SI{8.9}{\\meter} on Mag~4,\n"
     "which is why Section~\\ref{sec:incremental} treats the floor as a conditioning\n"
     "device to be kept minimal rather than a tuning knob.",
     "The largest single effect is the process-noise floor: raising it from\n"
     "$10^{-20}\\mathbf{I}$ to $10^{-16}\\mathbf{I}$ costs \\SI{8.9}{\\meter} smoothed\n"
     "and \\SI{32.0}{\\meter} causal on Mag~4. The floor is worth being explicit\n"
     "about. The mechanization gives a horizontal position channel of order\n"
     "$10^{-31}\\,\\mathrm{rad}^2$ per step, so even the smaller floor exceeds it by\n"
     "some eleven orders of magnitude and is, numerically, the operative position\n"
     "process noise; at $10^{-16}\\mathbf{I}$ it injects a random walk of\n"
     "\\SI{1.1}{\\meter} over the lag, which is the scale of the degradation\n"
     "observed. It is therefore a setting to be kept as small as conditioning\n"
     "allows, and the two points reported here bound it from above rather than\n"
     "demonstrate a plateau."),
    # ---- decade wording -------------------------------------------------
    ("one and a half to two orders of magnitude across the \\SIrange{30}{300}{\\second}\n"
     "gain regime and identifies the weakest flight line.",
     "a factor of \\numrange{24}{82} across the \\SIrange{30}{300}{\\second} gain\n"
     "regime, of which a factor of six is dimension counting, and identifies the\n"
     "weakest flight line."),
    # ---- ablation lead-in: four columns ---------------------------------
    ("Table~\\ref{tab:ablation} varies each element of the configuration on line\n"
     "1007.06, one at a time. The estimator is insensitive at the metre level to the\n"
     "modelling parameters, while excessive numerical regularization through the\n"
     "process-noise floor produces a larger degradation.",
     "Table~\\ref{tab:ablation} varies each element of the configuration on line\n"
     "1007.06, one at a time. Each entry is the smoothed DRMS with the causal value\n"
     "in parentheses, and the two react differently: the smoothed output is\n"
     "insensitive at the metre level to every modelling parameter, whereas the\n"
     "causal output is not, so the sentences below quote the largest change across\n"
     "the four columns rather than the smoothed column alone."),
    ("metre. Slowing the compensation walk a hundredfold costs about two metres,\n"
     "consistent with a real installation state that drifts in flight; speeding it up\n"
     "costs little.",
     "metre. Slowing the compensation walk a hundredfold costs \\SI{2.1}{\\meter}\n"
     "smoothed but \\SI{13.7}{\\meter} causal, the largest modelling effect in the\n"
     "table and a direct measure of how much the filtering estimate depends on the\n"
     "compensation being free to move; speeding it up costs little smoothed and\n"
     "\\SI{9.8}{\\meter} causal on Mag~5."),
    ("Inflating the assumed measurement noise fivefold, or relinearizing every\n"
     "variable at every epoch, changes the estimate by less than half a metre; the\n"
     "latter confirms that iSAM2's selective relinearization gives up nothing here.",
     "Inflating the assumed measurement noise fivefold changes the smoothed\n"
     "estimate by under half a metre but costs \\SI{7.7}{\\meter} causal on Mag~5;\n"
     "relinearizing every variable at every epoch changes nothing measurable,\n"
     "confirming that iSAM2's selective relinearization gives up nothing here.\n"
     "One row improves on the reference: widening the FOGM disturbance prior\n"
     "tenfold gains \\SI{3.9}{\\meter} causal on Mag~4 and a few tenths elsewhere,\n"
     "which we read as the disturbance state absorbing a little of the map error\n"
     "the nominal $\\sigma_S=\\SI{3}{\\nano\\tesla}$ is too tight to admit. We keep\n"
     "the nominal value because it is the physical setting and the gain is within\n"
     "the spread of the other rows."),
    # ---- contribution 3 / V-B contradiction -----------------------------
    ("EKF+TL+NN lineage of~\\cite{gnadt2022,hager2026} is reproduced and\n"
     "        tuned for the cold start, and the recursive EKF is given the\n"
     "        proposed estimator's own compensation prior, yet the smoother beats\n"
     "        both with no learned component.",
     "EKF+TL+NN model of~\\cite{gnadt2022} is reproduced and tuned for the\n"
     "        cold start, and the recursive EKF is given the proposed estimator's\n"
     "        own compensation prior, yet the smoother beats both with no learned\n"
     "        component."),
    # ---- units of B_k ----------------------------------------------------
    ("optionally augmented by a constant term, giving $m\\le19$",
     "with $B_k$ in nanotesla, optionally augmented by a constant term, giving $m\\le19$"),
]

miss = 0
for old, new in PAIRS:
    if old in src:
        src = src.replace(old, new)
    else:
        miss += 1
        print("MISS:", " ".join(old.split())[:76])
io.open(TEX, "w", encoding="utf-8").write(src)
print("round-6 pairs applied, misses:", miss)
