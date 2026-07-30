#!/usr/bin/env python3
"""Promote the separability condition into the methodology, add the missing
window-matrix definitions, deepen the fixed-lag mathematics, and rename the
breadth subsection.

Run: python paper/apply_promote_obs.py
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "taes_incremental.tex")

OBS_SUBSECTION = r"""\subsection{Separability of position and compensation}\label{sec:obs}
The competition just described can be made exact. Collect, over a window of $W$
epochs, the position sensitivities and the regressor rows of the linearized
measurement Eq.~\eqref{eq:measlin} into
\begin{equation}
\mathbf{G}=\begin{bmatrix}\mathbf{g}_1^{\top}\\ \vdots\\ \mathbf{g}_W^{\top}\end{bmatrix}
\in\mathbb{R}^{W\times2},
\qquad
\boldsymbol{\Psi}=\begin{bmatrix}\mathbf{A}_1^{\top}\\ \vdots\\ \mathbf{A}_W^{\top}\end{bmatrix}
\in\mathbb{R}^{W\times m},
\label{eq:gpsi}
\end{equation}
so that a constant position perturbation $\delta\mathbf{p}$ and a constant
coefficient perturbation $\delta\boldsymbol{\beta}$ change the residual
sequence by $\mathbf{G}\,\delta\mathbf{p}$ and
$\boldsymbol{\Psi}\,\delta\boldsymbol{\beta}$ respectively. The two unknowns
are distinguishable over the window exactly when no such pair produces the
same residual sequence.

\begin{proposition}\label{prop:obs}
A position perturbation $\delta\mathbf{p}$ is indistinguishable from a
compensation adjustment, hence unobservable from map matching over the
window, if and only if $\mathbf{G}\,\delta\mathbf{p}\in\mathrm{range}(\boldsymbol{\Psi})$.
Consequently the joint state $(\delta\mathbf{p},\delta\boldsymbol{\beta})$ is
observable if and only if the stacked matrix
$[\,\mathbf{G}\;\;\boldsymbol{\Psi}\,]$ has full column rank; equivalently, if and
only if all three of the following hold:
(i) $\ker\mathbf{G}=\{\mathbf{0}\}$ (the map gradient excites every
position direction over the window);
(ii) $\ker\boldsymbol{\Psi}=\{\mathbf{0}\}$ (the compensation basis has
full column rank); and
(iii)
$\mathrm{range}(\mathbf{G})\cap\mathrm{range}(\boldsymbol{\Psi})=\{\mathbf{0}\}$
(no shared direction). Condition (iii) alone governs the
position--compensation confounding: when it fails, a position shift with
$\mathbf{G}\,\delta\mathbf{p}\neq\mathbf{0}$ is reproduced exactly by a
compensation change.
\end{proposition}

The proof is in Appendix~\ref{app:proof}. The condition reads directly onto
the estimator design. The map gradient $\mathbf{g}_k$ follows the terrain
under the flight path while the regressor $\mathbf{A}_k$ follows the aircraft
attitude, so over a long enough window the two ranges decorrelate and
condition~(iii) is generically satisfied; over a single epoch ($W=1$) it
never is, which is the formal statement of why a memoryless correction cannot
work and why the lag of the smoother is the quantity that buys separability.

\begin{remark}\label{rem:floor}
The condition is geometric: it depends on the ranges of $\mathbf{G}$ and
$\boldsymbol{\Psi}$ over the window, hence on the flight path and the local map
gradient. It explains the dominant failure mode seen in practice: over a locally
flat map $\mathbf{G}\!\approx\!\mathbf{0}$ and condition (i) fails, so position is
unobservable regardless of the compensation, a map-information floor that bounds
every estimator.
\end{remark}

"""

FIXEDLAG_MATH = r"""The two outputs are the two classical posteriors. With
$\Delta$ the state interval and $\ell=L/\Delta$ the lag in states, the causal
estimate at epoch $k$ and the smoothed estimate for epoch $k-\ell$ are
\begin{equation}
\hat{\boldsymbol{\chi}}_{k|k}=\arg\max\,
p(\boldsymbol{\chi}_k\mid z_{0:k}),
\qquad
\hat{\boldsymbol{\chi}}_{k-\ell|k}=\arg\max\,
p(\boldsymbol{\chi}_{k-\ell}\mid z_{0:k}),
\label{eq:twoposteriors}
\end{equation}
the filtering and fixed-lag smoothing posteriors of the same graph.
Marginalization realizes the boundary compactly: if $\mathbf{x}_m$ denotes the
states leaving the lag and $\mathbf{x}_b$ the retained states their factors
touch, with joint information matrix and vector partitioned as
$\boldsymbol{\Lambda}=\big[\begin{smallmatrix}\boldsymbol{\Lambda}_{mm}&\boldsymbol{\Lambda}_{mb}\\
\boldsymbol{\Lambda}_{bm}&\boldsymbol{\Lambda}_{bb}\end{smallmatrix}\big]$,
$\boldsymbol{\eta}=\big[\begin{smallmatrix}\boldsymbol{\eta}_m\\
\boldsymbol{\eta}_b\end{smallmatrix}\big]$, then the marginal prior carried
forward is the Schur complement
\begin{equation}
\bar{\boldsymbol{\Lambda}}_{bb}
=\boldsymbol{\Lambda}_{bb}
-\boldsymbol{\Lambda}_{bm}\boldsymbol{\Lambda}_{mm}^{-1}\boldsymbol{\Lambda}_{mb},
\qquad
\bar{\boldsymbol{\eta}}_{b}
=\boldsymbol{\eta}_{b}
-\boldsymbol{\Lambda}_{bm}\boldsymbol{\Lambda}_{mm}^{-1}\boldsymbol{\eta}_{m},
\label{eq:schur}
\end{equation}
so no information is discarded, but the linearization at which
Eq.~\eqref{eq:schur} is evaluated is frozen: this is the precise sense in
which a smoothed state can no longer be revised.

"""


def main():
    src = io.open(TEX, encoding="utf-8").read()

    # 1. breadth heading
    src = src.replace(
        "\\subsection{Breadth}\\label{sec:breadth}",
        "\\subsection{Evaluation across lines and sensors}\\label{sec:breadth}")

    # 2. promote the appendix: drop it, insert the subsection before IV-B
    m = re.search(r"\\section\{Separability of Position and Compensation\}"
                  r".*?(?=\\section\{Pinson)", src, re.S)
    if not m:
        raise SystemExit("appendix block not found")
    src = src[:m.start()] + src[m.end():]
    anchor = "\\subsection{Factor graph formulation}\\label{sec:graphform}"
    src = src.replace(anchor, OBS_SUBSECTION + anchor)
    src = src.replace("is quantified in Appendix~\\ref{app:obs}.",
                      "is quantified in Section~\\ref{sec:obs}.")
    src = src.replace("Appendix~\\ref{app:obs}", "Section~\\ref{sec:obs}")

    # 3. richer fixed-lag mathematics, after the two-output itemize
    tail = ("Both are reported throughout Section~\\ref{sec:results}. Keeping "
            "them separate")
    if tail not in src:
        raise SystemExit("fixed-lag anchor not found")
    src = src.replace(tail, FIXEDLAG_MATH + tail)

    io.open(TEX, "w", encoding="utf-8").write(src)
    print("obs promoted, schur math added, breadth renamed")


if __name__ == "__main__":
    main()
