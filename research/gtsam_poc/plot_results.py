#!/usr/bin/env python3
"""Summary figure for the GTSAM PoC: trajectory + horizontal error timelines."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = 6378137.0
DT = 0.1

# Okabe-Ito (CVD-safe) fixed assignment
C_INS    = "#999999"
C_BATCH  = "#0072B2"   # offline bound
C_WINDOW = "#009E73"   # Julia-mirror sliding window
C_LAG300 = "#E69F00"   # per-epoch lag=300 smoothed
C_COLD   = "#D55E00"   # lag=30 realtime cold
C_WARM   = "#CC79A7"   # lag=30 realtime warm

d_batch = np.load("research/gtsam_poc/est_batch.npz")
d_win   = np.load("research/gtsam_poc/est_window.npz")
d_l300  = np.load("research/gtsam_poc/est_prog.npz")
d_cold  = np.load("research/gtsam_poc/est_prog_lag30.npz")
d_warm  = np.load("research/gtsam_poc/est_warm_lag30.npz")

tlat = d_batch["true_lat"]; tlon = d_batch["true_lon"]
ilat = d_batch["ins_lat"];  ilon = d_batch["ins_lon"]
N = tlat.size
t_min = np.arange(N) * DT / 60.0

lat0, lon0 = tlat.mean(), tlat.mean()*0 + tlon.mean()
def to_xy(lat, lon):
    return ((lon - lon0) * R * np.cos(tlat.mean()) / 1000.0,
            (lat - lat0) * R / 1000.0)

def herr(lat, lon):
    dn = (lat - tlat) * R
    de = (lon - tlon) * R * np.cos(tlat)
    return np.hypot(dn, de)

def est_ll(d, rt=False):
    if rt and "rt_lat" in d:
        return d["rt_lat"], d["rt_lon"]
    if rt and "est_rt" in d:
        return ilat + d["est_rt"][:, 0], ilon + d["est_rt"][:, 1]
    if "est_lat" in d:
        return d["est_lat"], d["est_lon"]
    return ilat + d["est"][:, 0], ilon + d["est"][:, 1]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
fig.subplots_adjust(left=0.05, right=0.99, bottom=0.12, top=0.88, wspace=0.24)

# --- (a) trajectory ---
ax = axes[0]
xt, yt = to_xy(tlat, tlon)
xi, yi = to_xy(ilat, ilon)
xw, yw = to_xy(*est_ll(d_win))
ax.plot(xi, yi, color=C_INS, lw=2, label="INS")
ax.plot(xw, yw, color=C_WINDOW, lw=2, label="FGO sliding window")
ax.plot(xt, yt, color="black", lw=1.2, ls="--", label="Truth")
ax.plot(xt[0], yt[0], "o", color="black", ms=6)
ax.annotate("start", (xt[0], yt[0]), textcoords="offset points",
            xytext=(6, 6), fontsize=9)
ax.set_xlabel("East [km]"); ax.set_ylabel("North [km]")
ax.set_title("(a) Trajectory — line 1007.06 (10 min, mag_5_uc)")
ax.legend(frameon=False, fontsize=9)
ax.set_aspect("equal")
ax.grid(alpha=0.25, lw=0.5)

# --- (b) horizontal error, smoothed/offline family ---
ax = axes[1]
ax.plot(t_min, herr(ilat, ilon), color=C_INS, lw=2, label="INS")
ax.plot(t_min, herr(*est_ll(d_l300)), color=C_LAG300, lw=2,
        label="per-epoch lag=300 s (smoothed)")
ax.plot(t_min, herr(*est_ll(d_win)), color=C_WINDOW, lw=2,
        label="sliding window ×3 (Julia mirror)")
ax.plot(t_min, herr(*est_ll(d_batch)), color=C_BATCH, lw=2,
        label="full batch (offline bound)")
ax.set_xlabel("time [min]"); ax.set_ylabel("horizontal error [m]")
ax.set_title("(b) Smoothed / offline estimates")
ax.legend(frameon=False, fontsize=9)
ax.set_ylim(0, 100)
ax.grid(alpha=0.25, lw=0.5)

# --- (c) realtime lag=30, cold vs warm ---
ax = axes[2]
ax.plot(t_min, herr(ilat, ilon), color=C_INS, lw=2, label="INS")
ax.plot(t_min, herr(*est_ll(d_cold, rt=True)), color=C_COLD, lw=2,
        label="lag=30 s realtime, cold TL")
ax.plot(t_min, herr(*est_ll(d_warm, rt=True)), color=C_WARM, lw=2,
        label="lag=30 s realtime, warm TL")
ax.axvline(5.0, color="black", lw=0.8, ls=":", alpha=0.6)
ax.annotate("warm-up 300 s", (5.0, 138), fontsize=8.5, ha="center",
            va="top", rotation=90, alpha=0.75)
ax.set_xlabel("time [min]"); ax.set_ylabel("horizontal error [m]")
ax.set_title("(c) Real-time (causal) estimates, per-epoch ISAM2")
ax.legend(frameon=False, fontsize=9)
ax.set_ylim(0, 140)
ax.grid(alpha=0.25, lw=0.5)

for ax in axes:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("GTSAM MagNav PoC — uncompensated cabin magnetometer, cold start",
             fontsize=12, y=0.98)
out = "research/gtsam_poc/gtsam_poc_results.png"
fig.savefig(out, dpi=150)
print("wrote", out)
