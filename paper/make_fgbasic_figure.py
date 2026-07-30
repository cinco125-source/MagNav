#!/usr/bin/env python3
"""Generic factor-graph figure for Section II-B.

A short trajectory showing the three factor types used everywhere in this paper:
one unary prior, binary process factors along the chain, and unary measurement
factors. Deliberately problem-agnostic; the graph of the actual estimator, with
its second chain of compensation variables, is Fig. fig_graph in Section III.

Run: python paper/make_fgbasic_figure.py -> paper/figs/fig_fgbasic.pdf
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from fig_style import apply_style, COL_W, C_PROPOSED, C_ACCENT

apply_style()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

XS = [1.0, 2.6, 4.2, 5.8]          # variable node positions
YV = 1.05                          # variable row
YM = 0.20                          # measurement-factor row
R = 0.26

fig, ax = plt.subplots(figsize=(COL_W, 1.55))
ax.set_xlim(0.1, 7.0)
ax.set_ylim(-0.10, 1.75)
ax.axis("off")


def variable(cx, label):
    ax.add_patch(Circle((cx, YV), R, fc="#cfe0f3", ec="k", lw=1.0, zorder=4))
    ax.text(cx, YV, label, ha="center", va="center", fontsize=8.5, zorder=5)


def factor(cx, cy, fc="k", s=0.15):
    ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, fc=fc, ec="k",
                           lw=0.6, zorder=4))


def edge(x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color="k", lw=0.9, zorder=2)


for i, cx in enumerate(XS):
    variable(cx, r"$\boldsymbol{\chi}_{%d}$" % i)
    # measurement factor hanging below each state
    factor(cx, YM, fc=C_PROPOSED)
    edge(cx, YV - R, cx, YM + 0.075)

# prior on the first state
factor(XS[0] - 0.62, YV, fc=C_ACCENT)
edge(XS[0] - 0.62 + 0.075, YV, XS[0] - R, YV)
ax.text(XS[0] - 0.62, YV + 0.30, "prior", ha="center", fontsize=7)

# process factors along the chain
for i in range(len(XS) - 1):
    fx = 0.5 * (XS[i] + XS[i + 1])
    factor(fx, YV)
    edge(XS[i] + R, YV, XS[i + 1] - R, YV)
ax.text(0.5 * (XS[0] + XS[1]), YV + 0.30, "process", ha="center", fontsize=7)
ax.text(XS[1], YM - 0.22, "measurement", ha="center", fontsize=7)

# continuation
ax.text(XS[-1] + 0.75, YV, r"$\cdots$", fontsize=12, va="center")
edge(XS[-1] + R, YV, XS[-1] + 0.55, YV)

fig.tight_layout(pad=0.15)
fig.savefig(os.path.join(OUT, "fig_fgbasic.pdf"))
print("wrote fig_fgbasic.pdf")
