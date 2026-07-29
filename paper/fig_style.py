#!/usr/bin/env python3
"""Shared figure style for the TAES manuscript.

Follows the Paper-Orchestra plot_style guide: Proposed (FGO) is always blue
solid, the primary baseline (EKF) is red, a reference/free-inertial trace is
black, serif fonts, grids, axis labels with units, vector PDF at >=300 dpi.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palette (MATLAB default set from the guide) -------------------------------
C_PROPOSED = "#0072BD"   # FGO  — proposed, always blue
C_BASE1    = "#D95319"   # EKF  — primary baseline, red
C_BASE2    = "#77AC30"   # secondary baseline, green
C_BASE3    = "#EDB120"   # tertiary, yellow
C_REF      = "#000000"   # reference / free inertial, black
C_ACCENT   = "#7E2F8E"   # purple accent for schematics
GRIDGRAY   = "#B8B8B8"

# Column widths (IEEE 2-column): single = 3.5 in, double = 7.16 in
COL_W = 3.5


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "grid.color": GRIDGRAY,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "axes.axisbelow": True,
    })


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
    ax.tick_params(length=2.5, width=0.7)
