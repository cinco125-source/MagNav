#!/usr/bin/env python3
"""Error-history figure for the representative line (Section V-C).

Line 1007.06 Mag 4 at the reference configuration: free-inertial drift, the
causal estimate, and the committed estimate, from
research/gtsam_poc/est_gtsam_ps100_1007_06_m4.npz. Nothing synthetic.

Run: python paper/make_errhist_figure.py -> paper/figs/fig_errhist.pdf
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from fig_style import apply_style, despine, COL_W, C_PROPOSED, C_BASE1, C_REF

apply_style()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs")
G = os.path.join(HERE, "..", "research", "gtsam_poc")
R_EARTH = 6378137.0

d = np.load(os.path.join(G, "est_gtsam_ps100_1007_06_m4.npz"))
il, io_, tl, tn = d["ins_lat"], d["ins_lon"], d["true_lat"], d["true_lon"]


def herr(est):
    dn = (il + est[:, 0] - tl) * R_EARTH
    de = (io_ + est[:, 1] - tn) * R_EARTH * np.cos(tl)
    return np.hypot(dn, de)


e_com = herr(d["est"])
e_cau = herr(d["est_rt"])
e_ins = np.hypot((il - tl) * R_EARTH, (io_ - tn) * R_EARTH * np.cos(tl))
t = np.arange(e_com.size) / 60.0          # states at 1 Hz -> minutes

fig, ax = plt.subplots(figsize=(COL_W, 2.5))
ax.plot(t, e_ins, ":", color=C_REF, lw=1.1, label="free inertial (317.5 m)")
ax.plot(t, e_cau, "-", color=C_BASE1, lw=1.1, alpha=0.9,
        label="causal (42.7 m)")
ax.plot(t, e_com, "-", color=C_PROPOSED, lw=1.4,
        label="committed, 300 s lag (24.2 m)")
ax.axvspan(0, 10, color="0.5", alpha=0.08, zorder=0)
ax.text(5, 380, "warm-up", ha="center", fontsize=7, color="0.4")
ax.set_xlabel("time [min]")
ax.set_ylabel("horizontal error [m]")
ax.set_xlim(0, t[-1])
ax.set_ylim(0, 420)
ax.grid(True)
ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
despine(ax)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "fig_errhist.pdf"))
print("wrote fig_errhist.pdf")
