#!/usr/bin/env python3
"""Splice the cold-start prior sections into the TAES manuscript.

Replaces the body of "Real-time incremental smoothing", whose measurements were
taken at a 30 s lag and a 10 Hz state cadence, with the new cold-start prior
analysis and the 1 Hz / 300 s lag timing, keeping the complexity paragraph and
the operational-integration paragraph that are still correct. Idempotent: it
refuses to run twice.

Run: python paper/integrate_coldstart.py
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "taes_fgo_magnav.tex")
DRAFT = os.path.join(HERE, "draft_coldstart.tex")

KEEP_HEAD = r"""\subsection{Real-time incremental smoothing}
Each window is a sparse nonlinear least-squares problem of fixed size
$O\big(L_w(n+m)\big)$; the block-bidiagonal Jacobian of the two chains
(Fig.~\ref{fig:graph}) is factored in $O(L_w)$ by the square-root
solver~\cite{dellaert2006}, and the number of Gauss--Newton iterations is small
(typically $3$--$5$) because the problem is mildly nonlinear away from observability
collapse. The per-window cost is therefore constant in the line length and the
overall complexity is linear in the number of epochs, matching a filter's
asymptotic cost while retaining the look-ahead. Memory is bounded by the window,
not the flight. The one-time overhead relative to an EKF is the window
factorization and the IRLS reweighting, which in our unoptimized research
implementation runs comfortably faster than real time on the SGL sampling rate.

"""

TAIL_MARK = "These same properties make the estimator compatible with online, fixed-lag"


def main():
    tex = io.open(TEX, encoding="utf-8").read()
    if "sec:coldprior" in tex and "\\subsection{The compensation prior at cold start}" in tex:
        sys.exit("already integrated")
    draft = io.open(DRAFT, encoding="utf-8").read()
    # drop the staging header comments
    draft = draft[draft.index("\\subsection{"):]

    i0 = tex.index("\\subsection{Real-time incremental smoothing}")
    i1 = tex.index(TAIL_MARK)
    new = KEEP_HEAD + draft.rstrip() + "\n\n"
    tex = tex[:i0] + new + tex[i1:]
    io.open(TEX, "w", encoding="utf-8").write(tex)
    print(f"spliced {len(draft)} chars of new material; manuscript now "
          f"{len(tex)} chars")


if __name__ == "__main__":
    main()
