#!/usr/bin/env python3
"""Shorten every caption, dissolve the Problem environment, and prefix
equation references with Eq./Eqs./Equation.

Captions are located by walking BACKWARD from each \\label to the nearest
\\caption and brace-matching its argument, so a replacement can never span
across other floats (the failure mode of the regex attempt).

Run: python paper/apply_short_captions2.py
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

PROBLEM_NEW = r"""Formally, the problem this paper solves is the following:
given the scalar magnetometer measurements $z_{0:N}$, the fluxgate regressor
rows $\mathbf{A}_{0:N}$, the anomaly map $h$, the inertial mechanization, and a
prior $(\hat{\boldsymbol{\chi}}_0,\mathbf{P}_0)$ that assumes no prior
knowledge of the aircraft field, estimate the joint trajectory
$\boldsymbol{\chi}_{0:N}$ of inertial navigation errors $\mathbf{x}_k$ and
time-varying compensation coefficients $\boldsymbol{\beta}_k$, with a per-epoch
cost that does not grow with flight duration, so that the estimate is available
on board."""


def brace_group(s, i):
    """s[i] == '{'; return index just past the matching close brace."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced")


def main():
    src = io.open(TEX, encoding="utf-8").read()

    # ---- captions, anchored at their labels -----------------------------
    done = 0
    for lab, cap in CAPTIONS.items():
        anchor = "\\label{" + lab + "}"
        li = src.find(anchor)
        if li < 0:
            print(f"  label missing: {lab}")
            continue
        ci = src.rfind("\\caption{", 0, li)
        if ci < 0:
            print(f"  caption missing before: {lab}")
            continue
        cend = brace_group(src, ci + len("\\caption"))
        src = src[:ci] + "\\caption{" + cap + "}" + src[cend:]
        done += 1
    print(f"captions: {done}/{len(CAPTIONS)}")

    # ---- problem environment -> prose -----------------------------------
    m = re.search(r"\\begin\{problem\}.*?\\end\{problem\}", src, re.S)
    if m:
        src = src[:m.start()] + PROBLEM_NEW + src[m.end():]
        src = src.replace("Three properties of Problem~\\ref{prob:joint} shape",
                          "Three properties of this problem shape")
        src = src.replace("Problem~\\ref{prob:joint} is stated as maximum a "
                          "posteriori (MAP) inference",
                          "The joint problem of Section~\\ref{sec:statement} "
                          "is stated as maximum a posteriori (MAP) inference")
        src = src.replace("Problem~\\ref{prob:joint}", "the joint problem")
        print("problem environment dissolved")
    else:
        print("  problem environment not found")

    # ---- Eq. prefixes for equation references ---------------------------
    # ranges first
    src = re.sub(r"(?<![.\w])\\eqref\{([^}]*)\}--\\eqref\{([^}]*)\}",
                 r"Eqs.~\\eqref{\1}--\\eqref{\2}", src)
    src = src.replace("~Eqs.~\\eqref", " Eqs.~\\eqref")
    # protect already-prefixed forms
    src = src.replace("Eq.~\\eqref{", "@EQP@{").replace("Eqs.~\\eqref{",
                                                        "@EQSP@{")
    src = src.replace("~\\eqref{", " Eq.~\\eqref{")
    src = src.replace("\\eqref{", "Eq.~\\eqref{")
    src = src.replace("Eq.~Eq.~", "Eq.~")
    src = src.replace("@EQP@{", "Eq.~\\eqref{").replace("@EQSP@{",
                                                        "Eqs.~\\eqref{")
    # sentence-initial Eq. -> Equation
    src = re.sub(r"(?<=[.!?]\s)Eq\.~\\eqref", r"Equation~\\eqref", src)
    src = re.sub(r"(?m)^Eq\.~\\eqref", "Equation~\\\\eqref", src)
    io.open(TEX, "w", encoding="utf-8").write(src)
    n_eq = src.count("Eq.~\\eqref") + src.count("Eqs.~\\eqref") \
        + src.count("Equation~\\eqref")
    print(f"equation references prefixed: {n_eq}")


if __name__ == "__main__":
    main()
