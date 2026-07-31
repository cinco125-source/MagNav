#!/usr/bin/env python3
"""Round-5 external review: soften the ablation claim to exempt the numerical
process-noise floor, drop the metric's knee-prediction overclaim, promote the
separability condition into the contribution list and abstract, and stop
claiming the Rao-Blackwellized decomposition is exact."""
import io

TEX = "taes_incremental.tex"
src = io.open(TEX, encoding="utf-8").read()

PAIRS = [
    # ---- (6) "no delicate setting" ---------------------------------------
    ("an ablation over every element of the configuration finds metre-level\n"
     "sensitivity and no delicate setting.",
     "an ablation over every element of the configuration finds metre-level\n"
     "sensitivity to the modelling parameters, and the expected sensitivity to\n"
     "excessive numerical regularization."),
    ("an ablation over every element of the configuration finds metre-level\n"
     "        sensitivity and no delicate setting.",
     "an ablation over every element of the configuration finds metre-level\n"
     "        sensitivity to the modelling parameters, and the expected\n"
     "        sensitivity to excessive numerical regularization."),
    ("The ablation finds no delicate\nsetting: one configuration serves every "
     "line and both sensors, and the update",
     "The ablation finds no delicate modelling\nparameter, apart from the "
     "numerical process-noise floor that must be kept\nminimal: one "
     "configuration serves every line and both sensors, and the update"),
    # ---- (2) knee prediction ---------------------------------------------
    ("The knee is anticipated by the\nseparability metric of Eq.~\\eqref{eq:mu}:",
     "The knee marks a change of regime, and the\nseparability metric of "
     "Eq.~\\eqref{eq:mu} identifies which one:"),
    # ---- (5) MPF "fits exactly" ------------------------------------------
    ("Since the TL model is linear\nin its coefficients, $\\boldsymbol{\\beta}$ "
     "is conditionally linear-Gaussian and\nfits the Rao--Blackwellized "
     "structure exactly: our implementation carries",
     "Since the TL model is linear\nin its coefficients, $\\boldsymbol{\\beta}$ "
     "is conditionally linear-Gaussian, which\nmotivates a "
     "Rao--Blackwellized implementation carrying the compensation and\nthe "
     "remaining approximately linear states in conditional Kalman blocks: "
     "ours carries"),
]

miss = 0
for old, new in PAIRS:
    if old in src:
        src = src.replace(old, new)
    else:
        miss += 1
        print("MISS:", " ".join(old.split())[:72])
io.open(TEX, "w", encoding="utf-8").write(src)
print("round-5 pairs applied, misses:", miss)
