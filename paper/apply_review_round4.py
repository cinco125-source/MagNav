#!/usr/bin/env python3
"""Fourth review round: the proposition becomes position identifiability
modulo the compensation gauge (the 19-term basis is verifiably rank-deficient
on the data), the two-output equation becomes an honest joint-MAP component
definition, the overclaims soften, the Huber notation unifies, the timing
counts drop the set-aside line, and the MPF leaves the comparison tables to
match what the text says about it.

Run: python paper/apply_review_round4.py
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "taes_incremental.tex")

PROP_NEW = r"""\begin{proposition}[Position separability modulo the compensation gauge]\label{prop:obs}
A position perturbation $\delta\mathbf{p}\neq\mathbf{0}$ is distinguishable
from every compensation adjustment over the window if and only if
$\mathbf{G}\,\delta\mathbf{p}\notin\mathrm{range}(\boldsymbol{\Psi})$;
equivalently, horizontal position is locally identifiable, modulo
$\ker\boldsymbol{\Psi}$, if and only if
\begin{equation}
\operatorname{rank}\!\big[(\mathbf{I}-\boldsymbol{\Psi}\boldsymbol{\Psi}^{\dagger})\,\mathbf{G}\big]=2,
\label{eq:projrank}
\end{equation}
where $\dagger$ denotes the Moore--Penrose pseudoinverse.
\end{proposition}

The proof is in Appendix~\ref{app:proof}. The condition deliberately does not
ask for identifiability of $\boldsymbol{\beta}$ itself, which would require
$\ker\boldsymbol{\Psi}=\{\mathbf{0}\}$, because the \num{19}-term basis does
not satisfy it: in continuous time
$B_t(u\dot{u}+v\dot{v}+w\dot{w})=0$ ties the three diagonal eddy columns
exactly, and $B_t(u^2+v^2+w^2)=B_t$ leaves the induced diagonal within about
one percent of the constant bias column, since the total field varies little
over a line. On the data, the smallest singular values of a
\SI{300}{\second} window of $\boldsymbol{\Psi}$ are six orders of magnitude
below the largest. Navigation, however, needs the compensation signal
$\boldsymbol{\Psi}\boldsymbol{\beta}$, not the coefficients: two coefficient
vectors differing by an element of $\ker\boldsymbol{\Psi}$ produce the same
correction, so the gauge is harmless to position, and
Eq.~\eqref{eq:projrank} is the condition that matters.

The condition also reads directly onto the estimator design. At a single
epoch, $\mathbf{G}$ has one row and the projected matrix has rank at most one,
so Eq.~\eqref{eq:projrank} can never hold: a memoryless correction cannot
separate the two effects. As the window grows, the map gradient under the
flight path and the attitude-driven regressor decorrelate and the projected
gradient accumulates rank and strength; the smallest singular value of the
whitened projected gradient,
\begin{equation}
\mu(W)=\lambda_{\min}\!\Big(\bar{\mathbf{G}}^{\top}
(\mathbf{I}-\bar{\boldsymbol{\Psi}}\bar{\boldsymbol{\Psi}}^{\dagger})
\bar{\mathbf{G}}\Big),
\qquad
\bar{\mathbf{G}}=R^{-1/2}\mathbf{G},\;
\bar{\boldsymbol{\Psi}}=R^{-1/2}\boldsymbol{\Psi},
\label{eq:mu}
\end{equation}
is a computable, flight-dependent measure of that strength, and
$\mu(W)^{-1/2}$ predicts the weakest-direction position uncertainty a window
of length $W$ can support under a static-coefficient reading of the model.
Section~\ref{sec:lagdesign} evaluates $\mu$ on the flight lines against the
measured lag knee. This static-perturbation analysis is a
measurement-level statement; in the implemented estimator the compensation
walk $\mathbf{Q}^{\beta}$, the inertial process factors, and the priors all
contribute additional coupling, and the empirical ablation of
Section~\ref{sec:ablation} is the measure of their effect."""

PROOF_NEW = r"""\section{Proof of Proposition~\ref{prop:obs}}\label{app:proof}
Over the window, a position perturbation $\delta\mathbf{p}$ and a coefficient
perturbation $\delta\boldsymbol{\beta}$ change the stacked residual by
$\mathbf{G}\,\delta\mathbf{p}+\boldsymbol{\Psi}\,\delta\boldsymbol{\beta}$.
The perturbation $\delta\mathbf{p}$ is indistinguishable from a compensation
adjustment exactly when some $\delta\boldsymbol{\beta}$ cancels it, that is
when $\mathbf{G}\,\delta\mathbf{p}=-\boldsymbol{\Psi}\,\delta\boldsymbol{\beta}
\in\mathrm{range}(\boldsymbol{\Psi})$, which proves the first statement.
For the second, decompose
$\mathbf{G}\,\delta\mathbf{p}=\boldsymbol{\Psi}\boldsymbol{\Psi}^{\dagger}
\mathbf{G}\,\delta\mathbf{p}+(\mathbf{I}-\boldsymbol{\Psi}\boldsymbol{\Psi}^{\dagger})
\mathbf{G}\,\delta\mathbf{p}$: the first term always lies in
$\mathrm{range}(\boldsymbol{\Psi})$, so
$\mathbf{G}\,\delta\mathbf{p}\in\mathrm{range}(\boldsymbol{\Psi})$ if and only
if $(\mathbf{I}-\boldsymbol{\Psi}\boldsymbol{\Psi}^{\dagger})\mathbf{G}\,
\delta\mathbf{p}=\mathbf{0}$. Hence every nonzero $\delta\mathbf{p}$ is
distinguishable if and only if the projected matrix
$(\mathbf{I}-\boldsymbol{\Psi}\boldsymbol{\Psi}^{\dagger})\mathbf{G}$ has
trivial kernel, that is rank two. \hfill$\blacksquare$
"""

TWOPOST_NEW = r"""Both are components of one optimization. Writing
$J_k$ for the objective Eq.~\eqref{eq:objective} restricted to the retained
window at epoch $k$, the smoother computes the joint estimate and reads two
components of it,
\begin{equation}
\hat{\mathbf{X}}_k=\arg\min_{\mathbf{X}}J_k(\mathbf{X}),\qquad
\hat{\boldsymbol{\chi}}^{\mathrm{c}}_{k}=[\hat{\mathbf{X}}_k]_{k},\qquad
\hat{\boldsymbol{\chi}}^{\mathrm{s}}_{k-\ell}=[\hat{\mathbf{X}}_k]_{k-\ell},
\label{eq:twoposteriors}
\end{equation}
with $\ell=L/\Delta$ the lag in states. Under a linear-Gaussian
specialization these components coincide with the classical filtering and
fixed-lag smoothing estimates; in the nonlinear robust formulation they are
the current and delayed components of the fixed-lag joint MAP solution, and
any covariance attached to them is the Laplace approximation at the current
linearization and IRLS weights."""


PAIRS = [
    # ---- two-output equation block replaced (anchor: its lead-in) --------
    # handled by regex below
    # ---- claims softening ------------------------------------------------
    ("solve it on board with an incremental fixed-lag smoother built on iSAM2",
     "solve it with a real-time-capable incremental fixed-lag smoother built "
     "on iSAM2"),
    ("cost that does not grow with flight duration, so that the estimate is available\non board.",
     "cost that does not grow with flight duration, so that the estimate is\navailable in flight."),
    ("The lag is the single design variable, and we measure what it buys",
     "The lag is the principal architecture-level design variable, and we "
     "measure what it buys"),
    ("The lag is the estimator's single design variable, and Table~\\ref{tab:lag}",
     "The lag is the estimator's principal design variable, and Table~\\ref{tab:lag}"),
    ("\\emph{Online EKF+TL+NN.} The strongest published causal method for this\ndataset augments",
     "\\emph{Online EKF+TL+NN.} The strongest causal baseline we compare\nagainst augments"),
    ("look-ahead is spent. Against the strongest published causal method, an EKF",
     "look-ahead is spent. Against an NN-augmented online EKF baseline, an EKF"),
    # ---- five lines / ten cases counts ----------------------------------
    ("a line. On the five survey lines used in Section~\\ref{sec:results} the map value",
     "a line. On the four flight lines used in Section~\\ref{sec:results} the map value"),
    ("\\SIrange{1729}{2081}{\\nano\\tesla} root-mean-square on\nMag~4 and",
     "\\SIrange{1729}{1978}{\\nano\\tesla} root-mean-square on\nMag~4 and"),
    ("ten line-magnetometer cases, mean \\SI{7.8}{\\milli\\second}, so the",
     "eight counted line--magnetometer cases, mean \\SI{8.2}{\\milli\\second},\nso the"),
    ("takes \\SIrange{6.0}{9.0}{\\milli\\second} per state across the",
     "takes \\SIrange{7.3}{9.0}{\\milli\\second} per state across the"),
    ("\\SIrange{6.0}{9.0}{\\milli\\second} per update.",
     "\\SIrange{7.3}{9.0}{\\milli\\second} per update."),
    ("(\\SIrange{6.0}{9.0}{\\milli\\second} per state at a \\SI{300}{\\second} lag and",
     "(\\SIrange{7.3}{9.0}{\\milli\\second} per state at a \\SI{300}{\\second} lag and"),
    ("runs a \\SI{300}{\\second} lag in real time at\n\\SIrange{6.0}{9.0}{\\milli\\second} per state.",
     "runs a \\SI{300}{\\second} lag in real time at\n\\SIrange{7.3}{9.0}{\\milli\\second} per state."),
    # ---- Huber notation: kernel of the whitened residual, not its square -
    ("+\\mathbf{A}_k^{\\top}\\boldsymbol{\\beta}_k+S_k-z_k\\big\\rVert^{2}_{R}\\Big),",
     "+\\mathbf{A}_k^{\\top}\\boldsymbol{\\beta}_k+S_k-z_k\\big\\rVert_{R}\\Big),"),
    # ---- conditioning floor wording --------------------------------------
    ("which injects under a millimetre of position uncertainty per\nstep and is dynamically negligible, and a QR factorization of the square-root",
     "which injects under a millimetre of position uncertainty per\nstep, and a QR factorization of the square-root"),
    # ---- MPF leaves Table III --------------------------------------------
    ("MPF{+}TL (best tuning) & 1227.0 & 75.1 & none \\\\\n", ""),
]


def main():
    src = io.open(TEX, encoding="utf-8").read()

    # 1. proposition block: from \begin{proposition} through the paragraph
    #    that ends before \begin{remark} (rem:floor stays, reworded below)
    m = re.search(r"\\begin\{proposition\}.*?\\end\{proposition\}", src, re.S)
    if not m:
        raise SystemExit("proposition not found")
    # remove the old post-proposition design paragraph up to rem:floor
    tail = src.index("\\begin{remark}", m.end())
    src = src[:m.start()] + PROP_NEW + "\n\n" + src[tail:]
    # the old lead-in above the proposition promised the three-condition form;
    # soften it
    src = src.replace(
        "so that a constant position perturbation $\\delta\\mathbf{p}$ and a constant\n"
        "coefficient perturbation $\\delta\\boldsymbol{\\beta}$ change the residual\n"
        "sequence by $\\mathbf{G}\\,\\delta\\mathbf{p}$ and\n"
        "$\\boldsymbol{\\Psi}\\,\\delta\\boldsymbol{\\beta}$ respectively. The two unknowns\n"
        "are distinguishable over the window exactly when no such pair produces the\n"
        "same residual sequence.",
        "so that a constant position perturbation $\\delta\\mathbf{p}$ and a constant\n"
        "coefficient perturbation $\\delta\\boldsymbol{\\beta}$ change the residual\n"
        "sequence by $\\mathbf{G}\\,\\delta\\mathbf{p}$ and\n"
        "$\\boldsymbol{\\Psi}\\,\\delta\\boldsymbol{\\beta}$ respectively.")

    # 2. rem:floor reword (condition (i) no longer exists)
    src = re.sub(
        r"\\begin\{remark\}\\label\{rem:floor\}.*?\\end\{remark\}",
        "\\\\begin{remark}\\\\label{rem:floor}\n"
        "The condition is geometric: it depends on the map gradient under the\n"
        "flight path. Over a locally flat map $\\\\mathbf{G}\\\\!\\\\approx\\\\!\\\\mathbf{0}$\n"
        "and the projected gradient of Eq.~\\\\eqref{eq:projrank} loses rank, so\n"
        "position is unobservable regardless of the compensation: a\n"
        "map-information floor that bounds every estimator.\n"
        "\\\\end{remark}", src, count=1, flags=re.S)

    # 3. proof rewritten
    src = re.sub(r"\\section\{Proof of Proposition~\\ref\{prop:obs\}\}\\label\{app:proof\}"
                 r".*?(?=\\bibliographystyle|\\section|\\end\{document\})",
                 PROOF_NEW.replace("\\", "\\\\") + "\n\n", src, count=1,
                 flags=re.S)

    # 4. two-output block
    m = re.search(r"The two outputs are the two classical posteriors\..*?"
                  r"in\s+which a smoothed state can no longer be revised\.",
                  src, re.S)
    if not m:
        raise SystemExit("two-posterior block not found")
    keep_schur = src[m.start():m.end()]
    schur_i = keep_schur.index("Marginalization realizes the boundary")
    src = src[:m.start()] + TWOPOST_NEW + "\n" + keep_schur[schur_i:] \
        + src[m.end():]

    # 5. simple pairs
    missed = []
    for old, new in PAIRS:
        if old in src:
            src = src.replace(old, new)
        else:
            missed.append(old)
    for mstr in missed:
        print(f"  NOT FOUND: {' '.join(mstr.split())[:70]}")

    # 6. MPF column leaves Table IV (7 -> 6 columns)
    src = src.replace(
        " & & EKF & EKF{+} & MPF{+} & \\multicolumn{2}{c}{Proposed} \\\\",
        " & & EKF & EKF{+} & \\multicolumn{2}{c}{Proposed} \\\\")
    src = src.replace(
        "Line & Mag & online & TL{+}NN & TL & causal & \\SI{300}{\\second} lag \\\\",
        "Line & Mag & online & TL{+}NN & causal & \\SI{300}{\\second} lag \\\\")
    src = src.replace("\\begin{tabular}{llrrr rr}", "\\begin{tabular}{llrr rr}")
    for row_old, row_new in [
        ("1007.06 & 4 & 46.7 & 48.8 & 1227.0 & 42.7 & \\textbf{24.2} \\\\",
         "1007.06 & 4 & 46.7 & 48.8 & 42.7 & \\textbf{24.2} \\\\"),
        ("1007.06 & 5 & 17.8 & 18.3 & 75.1 & 16.0 & \\textbf{11.5} \\\\",
         "1007.06 & 5 & 17.8 & 18.3 & 16.0 & \\textbf{11.5} \\\\"),
        ("1007.02 & 4 & div. & 130.0 & 1311.5 & 74.3 & \\textbf{28.4} \\\\",
         "1007.02 & 4 & div. & 130.0 & 74.3 & \\textbf{28.4} \\\\"),
        ("1007.02 & 5 & 31.6 & 30.4 & 2247.3 & 33.1 & \\textbf{15.8} \\\\",
         "1007.02 & 5 & 31.6 & 30.4 & 33.1 & \\textbf{15.8} \\\\"),
        ("1003.02 & 4 & div. & 101.0 & 482.6 & 58.2 & \\textbf{19.6} \\\\",
         "1003.02 & 4 & div. & 101.0 & 58.2 & \\textbf{19.6} \\\\"),
        ("1003.02 & 5 & 28.1 & 29.5 & 40.9 & 21.9 & \\textbf{12.0} \\\\",
         "1003.02 & 5 & 28.1 & 29.5 & 21.9 & \\textbf{12.0} \\\\"),
        ("1003.08 & 4 & err. & 46.6 & 2898.7 & 48.9 & \\textbf{25.0} \\\\",
         "1003.08 & 4 & err. & 46.6 & 48.9 & \\textbf{25.0} \\\\"),
        ("1003.08 & 5 & 21.1 & 20.7 & 77.1 & 19.2 & \\textbf{10.5} \\\\",
         "1003.08 & 5 & 21.1 & 20.7 & 19.2 & \\textbf{10.5} \\\\"),
    ]:
        if row_old in src:
            src = src.replace(row_old, row_new)
        else:
            print(f"  ROW NOT FOUND: {row_old[:44]}")
    src = src.replace("\\multicolumn{5}{l}{best causal}",
                      "\\multicolumn{4}{l}{best causal}")
    src = src.replace("\\multicolumn{5}{l}{best overall}",
                      "\\multicolumn{4}{l}{best overall}")

    io.open(TEX, "w", encoding="utf-8").write(src)
    print("round-4 surgery applied")


if __name__ == "__main__":
    main()
