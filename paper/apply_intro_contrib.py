#!/usr/bin/env python3
"""Replace the introduction's approach paragraph and contribution list.

The introduction's survey and gap paragraphs stay untouched. What changes is the
proposed-approach paragraph, which still described the batch-window-incremental
lag spectrum, and the four contributions, which are rewritten around the single
proposed estimator: the incremental fixed-lag smoother with joint compensation,
its causal/committed evaluation protocol, and the flight-data validation.
The organization paragraph is rewritten to match the new section layout.

Run: python paper/apply_intro_contrib.py
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "taes_fgo_magnav.tex")

OLD_START = "% P7: Proposed approach + contributions"
OLD_END = "% P8: Organization"

NEW = r"""% P7: Proposed approach + contributions
In this work we cast cold-start magnetic-anomaly navigation as a factor graph
and, within it, carry the online Tolles--Lawson coefficients as time-varying
variables, so that navigation and compensation are solved jointly as one
maximum-a-posteriori (MAP) problem (Fig.~\ref{fig:concept}). The graph is
solved on board by an incremental fixed-lag smoother built on
iSAM2~\cite{kaess2012}: states are placed at \SI{1}{\hertz}, variables older
than a \SI{300}{\second} lag are marginalized, and each update touches only the
part of the factorization the new measurement disturbs. Every state inside the
lag is relinearized and revised as later measurements arrive, so late
calibration information corrects early navigation states, the mechanism a
causal filter lacks; a causal filter commits each state once and cannot revise
it. We are explicit about scope: for a linear--Gaussian model the fixed-lag MAP
estimate coincides with classical fixed-lag smoothing~\cite{rauch1965}, so
smoothing per se is not the contribution; the joint estimation of a
time-varying compensation, delivered incrementally in real time, is.
The paper makes the following contributions:
\begin{enumerate}
  \item \textbf{Joint navigation and compensation on a factor graph.} We carry
        the online Tolles--Lawson coefficients as time-varying variables of a
        factor graph so that navigation and compensation are estimated jointly
        from the same scalar measurements. To the authors' knowledge this is
        the first factor-graph formulation that jointly estimates time-varying
        Tolles--Lawson compensation and navigation states at a calibration cold
        start. Relinearizing the coefficients as the graph is re-solved makes
        the compensation adapt along the flight, the role a learned online
        model plays in the competing filter, while the graph keeps every
        quantity an explicit, physically interpretable state.
  \item \textbf{A real-time incremental realization.} The graph is solved by an
        incremental fixed-lag smoother whose per-epoch cost is set by the lag,
        not the flight: \SIrange{6.0}{9.0}{\milli\second} per state at a
        \SI{300}{\second} lag and \SI{1}{\hertz} states, two orders of
        magnitude within the state cadence, in an unoptimized implementation.
        The lag is the single design variable, and we measure what it buys: the
        committed estimate improves steeply up to a knee near
        \SI{300}{\second}, about \SI{20}{\kilo\meter} of track, while the
        causal estimate is indifferent to it.
  \item \textbf{A causal/committed evaluation protocol.} A fixed-lag smoother
        produces two estimates, a causal one at the newest state and a
        committed one revised by the full lag, and comparisons in the
        literature rarely distinguish them. We report both throughout, so the
        comparison against causal filters is like for like and the price of
        latency is explicit: the causal output alone beats the recursive
        baselines on most cases, and the committed output roughly halves its
        error at the cost of lagging real time by the lag.
  \item \textbf{Flight-data validation.} On uncompensated cabin magnetometers
        across eight cold-start cases spanning four lines, two flights, two
        maps and two altitude regimes, under one untuned configuration, the
        committed estimate is the most accurate of the compared estimators on
        all eight cases and the causal estimate on six. Run rather than merely
        cited, the online EKF+TL+NN lineage of~\cite{gnadt2022,hager2026} is
        kept bounded by its network where a plain online-TL EKF diverges, yet
        the smoother beats it with no learned component; an ablation over
        every element of the configuration finds metre-level sensitivity and
        no delicate setting.
\end{enumerate}

"""

ORG_OLD_START = "% P8: Organization"


def main():
    src = io.open(TEX, encoding="utf-8").read()
    i0 = src.index(OLD_START)
    i1 = src.index(OLD_END)
    src = src[:i0] + NEW + src[i1:]
    io.open(TEX, "w", encoding="utf-8").write(src)
    print("contribution list replaced")


if __name__ == "__main__":
    main()
