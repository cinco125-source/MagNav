#!/usr/bin/env python3
"""Round-6b: effect sizes and significance, the dimension-counting null of the
metric, the Cramer-Rao identity, and the nuisance directions Proposition 1
omits. All numbers recomputed from the reported tables."""
import io

TEX = "taes_incremental.tex"
src = io.open(TEX, encoding="utf-8").read()

PAIRS = [
    # ---- breadth text: effect size instead of bare counts ---------------
    ("eight counted cases, between \\SI{10.5}{} and \\SI{28.4}{\\meter}, its DRMS\n"
     "lower than that of the best causal baseline on the same case by factors of\n"
     "\\num{1.5} to \\num{5.2}. Read causally, the proposed estimator is the best "
     "causal method on\nsix of the eight;",
     "eight counted cases, between \\SI{10.5}{} and \\SI{28.4}{\\meter}, its DRMS\n"
     "lower than that of the better causal baseline on the same case by factors of\n"
     "\\numrange{1.5}{5.2}, a geometric mean of \\num{2.29} with a 95\\% interval of\n"
     "\\numrange{1.65}{3.16} over the eight cases ($d_z=2.13$; sign test\n"
     "$p=0.004$). The causal reading is weaker and we state it as such: the\n"
     "proposed estimator is at least as accurate as the better causal baseline on\n"
     "six of the eight cases, a geometric-mean advantage of \\num{1.14} whose 95\\%\n"
     "interval \\numrange{0.96}{1.35} contains unity, so with eight cases the causal\n"
     "advantage is not statistically established (sign test $p=0.14$);"),
    # ---- conclusion ------------------------------------------------------
    ("free-inertial drift is hundreds of metres, reducing DRMS by factors of\n"
     "\\num{1.5} to \\num{5.2} relative to the best causal baseline.",
     "free-inertial drift is \\SIrange{121}{318}{\\meter}, reducing DRMS by a\n"
     "geometric-mean factor of \\num{2.29} (95\\% interval \\numrange{1.65}{3.16},\n"
     "$d_z=2.13$) relative to the better causal baseline. The causal margin is\n"
     "smaller and not statistically established on eight cases."),
    # ---- metric: separate the dimension-counting null --------------------
    ("the geography, and both match. The metric falls by one and a half to two\n"
     "orders of magnitude between \\SI{30}{} and \\SI{300}{\\second}, the range in\n"
     "which the measured smoothing gain is concentrated;",
     "the geography, and both match, once one correction is made. Part of the fall\n"
     "with window length is arithmetic rather than geometric: the residual space\n"
     "orthogonal to the $m=19$ compensation columns has $W-m$ dimensions, only\n"
     "\\num{11} at \\SI{30}{\\second} against \\num{281} at \\SI{300}{\\second}, and\n"
     "under an independent-noise null $\\bar{\\mathbf{G}}^{\\top}(\\mathbf{I}-\n"
     "\\bar{\\boldsymbol{\\Psi}}\\bar{\\boldsymbol{\\Psi}}^{\\dagger})\\bar{\\mathbf{G}}$\n"
     "is Wishart with $W-m$ degrees of freedom, which alone predicts a fall of\n"
     "\\num{6.1} with no geometry at all. The measured fall is \\numrange{24}{82},\n"
     "so the geometric contribution, the decorrelation of gradient from regressor,\n"
     "is a factor of \\numrange{3.9}{13.3}. That contribution is concentrated in\n"
     "the \\SIrange{30}{300}{\\second} range in which the measured smoothing gain\n"
     "lies;"),
    # ---- mu is the profiled Fisher information ---------------------------
    ("matrices, is a computable, flight-dependent measure of that strength:\n"
     "$\\mu(W)^{-1/2}$ is the weakest-direction position standard deviation that a\n"
     "single isolated window of $W$ epochs supports under a static-coefficient\n"
     "reading of the model.",
     "matrices, is a computable, flight-dependent measure of that strength. It is\n"
     "not an ad hoc quantity: for the linear model with independent noise it is\n"
     "the Schur complement of the Fisher information after the compensation is\n"
     "profiled out, so $\\mu(W)^{-1/2}$ is the weakest-direction Cram\\'er--Rao\n"
     "standard deviation of a single isolated window of $W$ epochs under a\n"
     "static-coefficient reading of the model."),
    ("It is not a bound on the estimator's error, and its\n"
     "assumptions cut both ways.",
     "It is a bound for that idealized single-window model and not for the\n"
     "smoother, and its assumptions cut both ways."),
]

miss = 0
for old, new in PAIRS:
    if old in src:
        src = src.replace(old, new)
    else:
        miss += 1
        print("MISS:", " ".join(old.split())[:76])

# ---- Remark on the nuisance directions Proposition 1 omits ---------------
anchor = "\\begin{remark}\\label{rem:floor}"
REMARK = (
    "\\begin{remark}\\label{rem:nuisance}\n"
    "The condition treats $\\delta\\mathbf{p}$ and $\\delta\\boldsymbol{\\beta}$ only.\n"
    "The window residual also carries the velocity and attitude errors and the\n"
    "disturbance state, and projecting those out as further nuisance directions\n"
    "can only reduce $\\mu$, so the metric is an upper bound on the information a\n"
    "window makes available. The velocity channel is the significant one: with the\n"
    "\\SI{1}{\\meter\\per\\second} prior of Table~\\ref{tab:params} the position ramp\n"
    "it generates spans up to \\SI{300}{\\meter} over a \\SI{300}{\\second} window,\n"
    "and over that window a ramp is not far from a constant offset. The\n"
    "disturbance state is partly absorbed already, its window-constant part lying\n"
    "in $\\mathrm{range}(\\boldsymbol{\\Psi})$ through the bias column, but with\n"
    "$\\tau=\\SI{180}{\\second}$ it decorrelates appreciably inside the window.\n"
    "\\end{remark}\n\n"
)
if anchor in src:
    src = src.replace(anchor, REMARK + anchor, 1)
else:
    miss += 1
    print("MISS: rem:floor anchor")

io.open(TEX, "w", encoding="utf-8").write(src)
print("round-6b applied, misses:", miss)
