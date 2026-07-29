#!/usr/bin/env python3
"""Cold-start compensation-prior figures for the TAES manuscript.

fig_prior.pdf  horizontal error against time on line 1007.06 Mag 4 for the
               per-epoch smoother at the nominal compensation prior and at a
               widened one, against free inertial.
fig_knee.pdf   position error of the first-W-second batch solution against W,
               for Mag 4 and Mag 5 of all five lines, at both priors.

Every series is read from a measured result file in research/gtsam_poc/;
nothing here is synthetic or interpolated. The .npz estimate arrays this reads
are too large to keep in the repository (see .gitignore); regenerate them with
research/gtsam_poc/breadth_prior.sh and prior_scale.sh, and the knee logs with
knee_sweep.sh and the sigma_beta=1000 variant, before running this script.

Run: wsl -e ~/gtsam_env/bin/python paper/make_coldstart_figures.py
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt

from fig_style import (apply_style, despine, COL_W,
                       C_PROPOSED, C_BASE1, C_REF)

apply_style()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)
G = os.path.join(HERE, "..", "research", "gtsam_poc")
R_EARTH = 6378137.0
DTK = 1.0                      # s per decimated state


def err_trace(npz):
    d = np.load(os.path.join(G, npz))
    est, il, io = d["est"], d["ins_lat"], d["ins_lon"]
    tl, tn = d["true_lat"], d["true_lon"]
    dn = (il + est[:, 0] - tl) * R_EARTH
    de = (io + est[:, 1] - tn) * R_EARTH * np.cos(tl)
    ins = np.hypot((il - tl) * R_EARTH, (io - tn) * R_EARTH * np.cos(tl))
    return np.arange(est.shape[0]) * DTK, np.hypot(dn, de), ins


# ---- fig_prior: nominal against widened compensation prior ----------------
t, e_tight, ins = err_trace("est_gtsam_dec300_1007_06_m4.npz")
_, e_wide, _ = err_trace("est_gtsam_ps100_1007_06_m4.npz")

fig, ax = plt.subplots(figsize=(COL_W, 2.5))
ax.semilogy(t / 60, ins, color=C_REF, ls=":", lw=1.2,
            label="free inertial (317 m)")
ax.semilogy(t / 60, e_tight, color=C_BASE1, lw=1.3,
            label=r"$\sigma_\beta=1$  (19457 m)")
ax.semilogy(t / 60, e_wide, color=C_PROPOSED, lw=1.5,
            label=r"$\sigma_\beta=100$  (24.2 m)")
ax.axvline(10, color="0.6", lw=0.8, ls="--")
ax.text(10.3, 3e4, "DRMS counted\nfrom here", fontsize=7, va="top")
ax.set_xlabel("time [min]")
ax.set_ylabel("horizontal error [m]")
ax.set_xlim(0, t[-1] / 60)
ax.set_ylim(1, 1e5)
ax.grid(True, which="both")
ax.legend(loc="lower right", framealpha=0.9)
despine(ax)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "fig_prior.pdf"))
print("wrote fig_prior.pdf")


# ---- fig_knee: first-window solution against window length ---------------
def parse(path):
    rows = {}
    line = mag = None
    for ln in open(path):
        m = re.match(r"#+ (\S+) #+", ln)
        if m:
            line = m.group(1).replace("_", ".")
        m = re.match(r"mag(\d) K=", ln)
        if m:
            mag = m.group(1)
        m = re.match(r"W=\s*(\d+).*drms\s+([\d.]+) m", ln)
        if m:
            rows.setdefault((line, mag), {})[int(m.group(1))] = float(m.group(2))
    return rows


tight = parse(os.path.join(G, "knee_sweep.log"))
wide = parse(os.path.join(G, "knee_prior.log"))

fig, ax = plt.subplots(figsize=(COL_W, 2.6))
for rows, color, ls in ((tight, C_BASE1, "-"), (wide, C_PROPOSED, "-")):
    for (line, mag), d in sorted(rows.items()):
        if mag != "4":
            continue
        W = sorted(d)
        ax.semilogy(W, [d[w] for w in W], ls, color=color, lw=1.1, alpha=0.8,
                    marker="o", ms=2.5)
ax.semilogy([], [], "-o", ms=2.5, color=C_BASE1, lw=1.1,
            label=r"$\sigma_\beta=1$")
ax.semilogy([], [], "-o", ms=2.5, color=C_PROPOSED, lw=1.1,
            label=r"$\sigma_\beta=1000$")
ax.set_xlabel(r"window length $W$ [s]")
ax.set_ylabel("error of window solution [m]")
ax.set_title("Mag 4, five lines", fontsize=9)
ax.grid(True, which="both")
ax.legend(loc="upper right", framealpha=0.9)
despine(ax)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "fig_knee.pdf"))
print("wrote fig_knee.pdf")
