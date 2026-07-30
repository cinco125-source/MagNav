#!/usr/bin/env python3
"""Split the preliminaries draft into two sections and add the problem statement.

Section II becomes Problem Formulation, ending with a formal statement of what
is estimated and under what conditions; the factor graph material becomes its
own Section III, so the reader meets the problem before the machinery.

Run: python paper/restructure_prelim.py
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "draft_prelim.tex")

OLD_HEAD = r"""\section{Preliminaries}\label{sec:prelim}
This section states the problem in its classical batch form and then the factor
graph machinery used to solve it, so that Section~\ref{sec:method} can put the
two together. The batch form makes clear what magnetic-anomaly navigation asks
for and why the aircraft's own field prevents it from being answered directly;
the factor graph form supplies the language in which the field and the
trajectory can be estimated at once."""

NEW_HEAD = r"""\section{Problem Formulation}\label{sec:problem}
This section states what magnetic-anomaly navigation asks for in its classical
batch form, shows why the aircraft's own field prevents that form from being
answered directly, and states the problem solved in the rest of the paper."""

OLD_FGO = r"""\subsection{Factor graph optimization}\label{sec:fgo}
The joint problem is stated as maximum a posteriori (MAP) inference, and solved
on a factor graph. This subsection sets out that machinery from the beginning,
since the structure of the graph is what later allows the compensation to be
carried as a first-class unknown and the estimate to be produced in real time."""

NEW_FGO = r"""\section{Factor Graph Optimization}\label{sec:fgo}
Problem~\ref{prob:joint} is stated as maximum a posteriori (MAP) inference and
solved on a factor graph. This section sets out that machinery from the
beginning, since the structure of the graph is what allows the compensation to
be carried as a first-class unknown and the estimate to be produced in real
time."""

PROBLEM = r"""
\subsection{Problem statement}\label{sec:statement}
The situation just described fixes what has to be estimated and what may be
assumed. The aircraft carries a strapdown inertial navigation system whose
error grows without bound, a three-axis fluxgate, and a scalar magnetometer
whose reading is corrupted by the aircraft field. A gridded anomaly map of the
overflown region is available. No calibration flight has been made, so no
compensation model is known at takeoff, and no satellite navigation is
available at any point.

\begin{problem}[Joint navigation and compensation at a cold start]\label{prob:joint}
Given the scalar magnetometer measurements $z_{0:N}$, the fluxgate regressor
rows $\mathbf{A}_{0:N}$, the anomaly map $h$, the inertial mechanization, and a
prior $(\hat{\boldsymbol{\chi}}_0,\mathbf{P}_0)$ that assumes no prior
knowledge of the aircraft field, estimate the joint trajectory
$\boldsymbol{\chi}_{0:N}$ of inertial navigation errors $\mathbf{x}_k$ and
time-varying compensation coefficients $\boldsymbol{\beta}_k$, and do so with a
per-epoch cost that does not grow with flight duration, so that the estimate is
available on board.
\end{problem}

Three properties of Problem~\ref{prob:joint} shape the rest of the paper. It is
\emph{joint}, because the compensation cannot be measured separately from the
position it corrupts. It is \emph{underdetermined at any single epoch}, because
one scalar reading cannot separate a position error from a compensation error,
so the two are told apart only over a span of epochs. And it is
\emph{online}, because an estimator that must see the whole flight before
answering is of no use to the aircraft that flew it. The first two properties
call for an estimator that carries the compensation as a state and revises past
states as later data arrive; the third bounds how far into the past it may keep
revising. Sections~\ref{sec:fgo} and~\ref{sec:method} construct such an
estimator.
"""


def main():
    src = io.open(SRC, encoding="utf-8").read()
    for old, new in ((OLD_HEAD, NEW_HEAD), (OLD_FGO, NEW_FGO)):
        if old not in src:
            raise SystemExit(f"anchor not found: {' '.join(old.split())[:60]}")
        src = src.replace(old, new)
    # the problem statement closes Section II, just before the factor graph section
    src = src.replace(NEW_FGO, PROBLEM.lstrip("\n") + "\n" + NEW_FGO)
    for a, b in (("Variables, factors, and the graph", None),
                 ("From factors to least squares", None),
                 ("Solving the graph", None),
                 ("Batch, fixed-lag, and incremental", None)):
        src = src.replace("\\subsubsection{" + a + "}", "\\subsection{" + a + "}")
    io.open(SRC, "w", encoding="utf-8").write(src)
    print("restructured")


if __name__ == "__main__":
    main()
