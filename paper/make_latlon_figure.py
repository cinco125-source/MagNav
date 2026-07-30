#!/usr/bin/env python3
"""Latitude/longitude error histories of every compared method (Section V-C).

Line 1007.06 Mag 4: north (latitude) and east (longitude) error against time
for the online-TL EKF, the EKF+TL+NN reference, and the proposed estimator's
causal and committed outputs. Baseline trajectories come from
research/baseline_tracks_1007_06_m4.csv (dump_baselines.jl); the proposed
trajectories from research/gtsam_poc/est_gtsam_ps100_1007_06_m4.npz.
The style follows the error-history figures of Park and Bang (IJCAS 2024).

Run: wsl -e ~/gtsam_env/bin/python paper/make_latlon_figure.py
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from fig_style import (apply_style, despine, COL_W,
                       C_PROPOSED, C_BASE1, C_BASE3)

apply_style()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs")
G = os.path.join(HERE, "..", "research")
R_EARTH = 6378137.0
C_WEAK = "#9aa4b2"

rows = list(csv.DictReader(open(os.path.join(G, "baseline_tracks_1007_06_m4.csv"))))
get = lambda k: np.array([float(r[k]) for r in rows])
t = get("t") / 60.0
tlat, tlon = get("true_lat"), get("true_lon")
coslat = np.cos(tlat)

def errs(lat, lon):
    return (lat - tlat) * R_EARTH, (lon - tlon) * R_EARTH * coslat

e_ekf = errs(get("ekf_lat"), get("ekf_lon"))
e_nn = errs(get("nn_lat"), get("nn_lon"))

d = np.load(os.path.join(G, "gtsam_poc", "est_gtsam_ps100_1007_06_m4.npz"))
tl1, tn1 = d["true_lat"], d["true_lon"]
il1, io1 = d["ins_lat"], d["ins_lon"]
cos1 = np.cos(tl1)
tk = np.arange(tl1.size) / 60.0

def perrs(est):
    return ((il1 + est[:, 0] - tl1) * R_EARTH,
            (io1 + est[:, 1] - tn1) * R_EARTH * cos1)

e_com = perrs(d["est"])
e_cau = perrs(d["est_rt"])

fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.2), sharex=True)
for ax, comp, name in zip(axes, (0, 1), ("north (latitude)", "east (longitude)")):
    ax.plot(t, e_ekf[comp], color=C_WEAK, lw=0.8, label="EKF, online TL")
    ax.plot(t, e_nn[comp], color=C_BASE1, lw=0.8, label="EKF+TL+NN")
    ax.plot(tk, e_cau[comp], color=C_BASE3, lw=0.8, label="proposed, causal")
    ax.plot(tk, e_com[comp], color=C_PROPOSED, lw=1.1,
            label="proposed, smoothed")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.axvspan(0, 10, color="0.5", alpha=0.08, zorder=0)
    ax.set_ylabel(f"{name.split()[0]} error [m]", fontsize=8)
    ax.set_ylim(-130, 130)
    ax.grid(True)
    despine(ax)
axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=2,
               fontsize=6.6, frameon=False, columnspacing=1.0,
               handlelength=1.3)
axes[1].set_xlabel("time [min]")
axes[1].set_xlim(0, t[-1])
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "fig_latlon.pdf"))
fig.savefig(os.path.join(OUT, "fig_latlon.png"), dpi=170)
print("wrote fig_latlon")
