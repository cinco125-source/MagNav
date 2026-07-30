#!/usr/bin/env python3
"""Shorten every caption to one or two lines and dissolve the numbered Problem
environment into prose (a single numbered problem adds machinery without
payoff). References to Problem 1 become references to the joint problem.

Run: python paper/apply_short_captions.py
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "taes_incremental.tex")

CAPTIONS = {
    "fig:concept": "Concept: one scalar reading couples position and "
                   "compensation; a joint fixed-lag factor graph estimates "
                   "both, emitting a causal and a smoothed answer.",
    "fig:signal": "Line 1007.06 Mag~4: the uncompensated aircraft field "
                  "exceeds the full range of the map signal along the track.",
    "fig:graph": "Factor graph maintained by the smoother: two chains coupled "
                 "at every epoch by the robust map factor carrying $z_k$; the "
                 "marginal prior summarizes states older than the lag. Relief "
                 "height is anomaly strength.",
    "fig:latlon": "North and east error histories, line 1007.06 Mag~4. The "
                  "seed-sensitive EKF+TL+NN is a fresh run "
                  "(\\SI{40.0}{\\meter} here).",
    "fig:errhist": "Horizontal error, line 1007.06 Mag~4: the smoothed trace "
                   "is the causal trace revised by the \\SI{300}{\\second} "
                   "lag.",
    "fig:track": "Line 1007.06 Mag~4: (a) flown line with the zoom window; "
                 "(b) the estimated paths separate from the truth, at the "
                 "largest post-warm-up EKF error.",
    "fig:comp": "Estimated $\\mathbf{A}^{\\top}\\boldsymbol{\\beta}+S$ "
                "against the interference reference, and the residual "
                "(line 1007.06 Mag~4).",
    "fig:breadthbar": "Breadth comparison of Table~\\ref{tab:breadth}, log "
                      "axis.",
    "fig:lagsweep": "Smoothed DRMS against lag; open markers show the "
                    "lag-independent causal error.",
    "tab:params": "Estimator settings, common to every case.",
    "tab:lines": "Flight lines (FF free flight, SV survey; maps R Renfrew, E "
                 "Eastern) and free-inertial drift.",
    "tab:1007": "Line 1007.06 cold start: DRMS [m] after a "
                "\\SI{600}{\\second} warm-up.",
    "tab:breadth": "Breadth: DRMS [m]. ``div.''\\ $>\\SI{10}{\\kilo\\meter}$, "
                   "``err.''\\ leaves the map; bold = best overall.",
    "tab:lag": "Lag sweep: smoothed DRMS [m]; the causal output varies by "
               "under a metre across the sweep.",
    "tab:ablation": "Ablation on line 1007.06, one change per row: smoothed "
                    "(causal) DRMS [m].",
}

PROBLEM_OLD_RE = re.compile(
    r"\\begin\{problem\}.*?\\end\{problem\}", re.S)

PROBLEM_NEW = r"""Formally, the problem this paper solves is the following:
given the scalar magnetometer measurements $z_{0:N}$, the fluxgate regressor
rows $\mathbf{A}_{0:N}$, the anomaly map $h$, the inertial mechanization, and a
prior $(\hat{\boldsymbol{\chi}}_0,\mathbf{P}_0)$ that assumes no prior
knowledge of the aircraft field, estimate the joint trajectory
$\boldsymbol{\chi}_{0:N}$ of inertial navigation errors $\mathbf{x}_k$ and
time-varying compensation coefficients $\boldsymbol{\beta}_k$, with a per-epoch
cost that does not grow with flight duration, so that the estimate is available
on board."""


def main():
    src = io.open(TEX, encoding="utf-8").read()
    n = 0
    for lab, cap in CAPTIONS.items():
        pat = re.compile(r"\\caption\{.*?\}\s*\n\\label\{" + re.escape(lab)
                         + r"\}", re.S)
        new = "\\\\caption{" + cap.replace("\\", "\\\\") + "}\n\\\\label{" \
              + lab + "}"
        src, k = pat.subn(new, src, count=1)
        n += k
        if not k:
            print(f"  caption not found: {lab}")
    src, k = PROBLEM_OLD_RE.subn(PROBLEM_NEW.replace("\\", "\\\\"), src,
                                 count=1)
    if not k:
        print("  problem environment not found")
    # references to the numbered problem
    src = src.replace("Three properties of Problem~\\ref{prob:joint} shape",
                      "Three properties of this problem shape")
    src = src.replace("Problem~\\ref{prob:joint} is stated as maximum a "
                      "posteriori (MAP) inference",
                      "The joint problem of Section~\\ref{sec:statement} is "
                      "stated as maximum a posteriori (MAP) inference")
    src = src.replace("Problem~\\ref{prob:joint}", "the joint problem")
    io.open(TEX, "w", encoding="utf-8").write(src)
    print(f"captions replaced: {n}/{len(CAPTIONS)}; "
          f"residual prob refs: {src.count('prob:joint')}")


if __name__ == "__main__":
    main()
