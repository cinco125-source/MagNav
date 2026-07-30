#!/usr/bin/env python3
"""Observability metric mu(L)^{-1/2} against the measured lag sweep.

Data sources:
  research/gtsam_poc/obs_metric_results.csv   (obs_metric.py: median and IQR of
      the whitened projected-gradient metric over sliding windows, per line)
  smoothed DRMS vs lag: paper/taes_incremental.tex tab:lag, produced by
      research/gtsam_poc/gtsam_poc_result_ps100_*.txt lag sweep runs.

One panel: dashed = mu^{-1/2} median with IQR band (metric, per line, shared by
both magnetometers); solid with markers = measured smoothed DRMS, Mag 4. Same
color per line. Output paper/figs/fig_obsmetric.pdf.
"""
import csv
import os

import matplotlib.pyplot as plt

from fig_style import COL_W, apply_style, despine

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "research", "gtsam_poc",
                   "obs_metric_results.csv")
OUT = os.path.join(HERE, "figs", "fig_obsmetric.pdf")

# tab:lag, Mag 4 rows (smoothed DRMS [m] at lag 30/60/150/300/600 s)
LAG_X = [30, 60, 150, 300, 600]
DRMS4 = {
    "1007_06": [40.2, 37.1, 29.7, 24.2, 21.9],
    "1007_02": [70.2, 60.4, 43.2, 28.4, 25.8],
    "1003_02": [48.9, 43.3, 28.8, 19.6, 18.2],
    "1003_08": [50.3, 41.5, 31.5, 25.0, 24.6],
}
COLORS = {"1007_06": "#0072BD", "1007_02": "#D95319",
          "1003_02": "#77AC30", "1003_08": "#7E2F8E"}
NAMES = {"1007_06": "1007.06", "1007_02": "1007.02",
         "1003_02": "1003.02", "1003_08": "1003.08"}


def main():
    apply_style()
    met = {}
    with open(CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            met.setdefault(r["line"], []).append(
                (int(r["lag_s"]), float(r["sig_med"]),
                 float(r["sig_q1"]), float(r["sig_q3"])))

    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    for line, rows in met.items():
        rows.sort()
        x = [r[0] for r in rows]
        c = COLORS[line]
        ax.plot(x, [r[1] for r in rows], ls="--", lw=1.3, color=c, zorder=3)
        ax.fill_between(x, [r[2] for r in rows], [r[3] for r in rows],
                        color=c, alpha=0.10, lw=0, zorder=1)
        ax.plot(LAG_X, DRMS4[line], ls="-", lw=1.5, color=c, marker="o",
                ms=3.4, zorder=4, label=NAMES[line])
    from matplotlib.ticker import NullFormatter
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([30, 60, 120, 300, 600])
    ax.set_xticklabels(["30", "60", "120", "300", "600"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("window / lag $L$ [s]")
    ax.set_ylabel("[m]")
    ax.grid(True, which="major")
    despine(ax)
    # legend below the axes: line colors + linestyle meaning
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=COLORS[k], lw=1.5, marker="o", ms=3.4,
                 label=NAMES[k]) for k in NAMES]
    hs += [Line2D([], [], color="0.3", lw=1.5, marker="o", ms=3.4,
                  label="smoothed DRMS (Mag 4)"),
           Line2D([], [], color="0.3", lw=1.3, ls="--",
                  label=r"$\mu(L)^{-1/2}$ median, IQR")]
    fig.legend(handles=hs, ncol=3, fontsize=6.8, frameon=False,
               loc="lower center", bbox_to_anchor=(0.54, -0.02),
               handlelength=1.8, columnspacing=0.8)
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(OUT)
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
