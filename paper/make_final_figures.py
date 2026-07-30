#!/usr/bin/env python3
"""Publication figures for taes_incremental.tex, drawn from measured data.

fig_concept   (a) anomaly map with the flown track; (b) the measured scalar
              against the map value along the track, showing the uncompensated
              aircraft field dwarfing the map signal. Line 1007.06 Mag 4.
fig_fgbasic   generic factor graph: prior, process, measurement factors.
fig_graph     the joint two-chain graph with map factors coupling the chains.
fig_track     map with true, INS and committed tracks, line 1007.06 Mag 4.
fig_comp      what the compensation chain learned: interference reference
              against the estimated A'beta + S, and the residual.
fig_breadthbar  breadth comparison as a log-scale bar chart.
fig_lagsweep  committed DRMS against lag, causal shown flat, per line.

Data: ~/magnav_data/line_1007_06_full.h5 and
research/gtsam_poc/est_gtsam_ps100_1007_06_m4.npz, plus the values of
Tables IV-V (gtsam_poc_result_ps100_*, _lagd*_*). Nothing synthetic.

Run: wsl -e ~/gtsam_env/bin/python paper/make_final_figures.py
"""
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

from fig_style import (apply_style, despine, COL_W,
                       C_PROPOSED, C_BASE1, C_BASE2, C_BASE3, C_REF, C_ACCENT)

apply_style()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs")
G = os.path.join(HERE, "..", "research", "gtsam_poc")
H5 = os.path.expanduser("~/magnav_data/line_1007_06_full.h5")
R_EARTH = 6378137.0
CMAP = "RdBu_r"

# ---------------------------------------------------------------- data
with h5py.File(H5, "r") as f:
    glat = np.rad2deg(f["glat"][()].ravel())
    glon = np.rad2deg(f["glon"][()].ravel())
    gh = f["gh_rowmajor"][()]
    tlat = f["true_lat"][()].ravel()
    tlon = f["true_lon"][()].ravel()
    ilat = f["ins_lat"][()].ravel()
    ilon = f["ins_lon"][()].ravel()
    meas4 = f["meas_mag4"][()].ravel()
    A = np.asarray(f["A"][()], float)
    dt = float(f["dt"][()])

anom = gh - gh.mean()                      # remove the core-field offset

def bilin(lat, lon):
    gla = np.deg2rad(glat); glo = np.deg2rad(glon)
    i = np.clip((lat - gla[0]) / (gla[1] - gla[0]), 0, gla.size - 1.001)
    j = np.clip((lon - glo[0]) / (glo[1] - glo[0]), 0, glo.size - 1.001)
    i0 = i.astype(int); j0 = j.astype(int)
    fi = i - i0; fj = j - j0
    return (gh[i0, j0] * (1 - fi) * (1 - fj) + gh[i0 + 1, j0] * fi * (1 - fj)
            + gh[i0, j0 + 1] * (1 - fi) * fj + gh[i0 + 1, j0 + 1] * fi * fj)

d = np.load(os.path.join(G, "est_gtsam_ps100_1007_06_m4.npz"))
est, est_rt = d["est"], d["est_rt"]
idx = d["idx"] if "idx" in d.files else np.arange(0, meas4.size, 10)[:est.shape[0]]
il1, io1 = d["ins_lat"], d["ins_lon"]
tl1, tn1 = d["true_lat"], d["true_lon"]

tdeg = lambda r: np.rad2deg(r)
bbox = dict(lat=(tdeg(tlat).min() - 0.02, tdeg(tlat).max() + 0.02),
            lon=(tdeg(tlon).min() - 0.02, tdeg(tlon).max() + 0.02))


def map_panel(ax, cbar=True):
    im = ax.imshow(anom, origin="lower", cmap=CMAP,
                   extent=[glon[0], glon[-1], glat[0], glat[-1]],
                   vmin=-np.percentile(np.abs(anom), 99),
                   vmax=np.percentile(np.abs(anom), 99),
                   aspect="auto", interpolation="bilinear")
    ax.set_xlim(*bbox["lon"]); ax.set_ylim(*bbox["lat"])
    ax.set_xlabel("longitude [deg]"); ax.set_ylabel("latitude [deg]")
    return im


# ================================================================ fig_concept
fig = plt.figure(figsize=(7.16, 2.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.55)

axm = fig.add_subplot(gs[0])
im = map_panel(axm)
axm.plot(tdeg(tlon), tdeg(tlat), color="k", lw=1.2)
axm.plot(tdeg(tlon[0]), tdeg(tlat[0]), marker="^", color="k", ms=6)
axm.annotate("start", (tdeg(tlon[0]), tdeg(tlat[0])),
             textcoords="offset points", xytext=(-26, 6), fontsize=7)
cb = fig.colorbar(im, ax=axm, fraction=0.052, pad=0.02)
cb.set_label("anomaly [nT]", fontsize=7, labelpad=1)
cb.ax.tick_params(labelsize=6)
axm.set_title("(a) anomaly map and flown track", fontsize=8.5)

axs = fig.add_subplot(gs[1])
t_min = np.arange(meas4.size) * dt / 60.0
h_track = bilin(tlat, tlon)
map_sig = h_track - h_track.mean()
intf = meas4 - h_track                      # uncompensated aircraft field
axs.plot(t_min, intf, color=C_BASE1, lw=0.7,
         label="aircraft field  $z-h(\\mathbf{p})$")
axs.plot(t_min, map_sig, color=C_PROPOSED, lw=0.8,
         label="map signal along track")
axs.axhline(0, color="0.6", lw=0.5)
axs.set_xlabel("time [min]"); axs.set_ylabel("[nT]")
axs.set_xlim(0, t_min[-1])
axs.legend(loc="center right", fontsize=7, framealpha=0.9)
axs.grid(True)
axs.set_title("(b) the interference exceeds the signal", fontsize=8.5)
despine(axs)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(OUT, "fig_concept.pdf"))
fig.savefig(os.path.join(OUT, "fig_concept.png"), dpi=170)
plt.close(fig)
print("fig_concept")

# =============================================================== graph helpers
FC_VAR_X = "#dbe7f5"      # inertial variables
FC_VAR_B = "#fdeecd"      # compensation variables
EC_VAR = "#37465a"
FC_PRIOR = C_ACCENT
FC_PROC = "#37465a"
FC_MEAS = C_PROPOSED


def gvar(ax, x, y, txt, fc, r=0.30):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=EC_VAR, lw=1.1, zorder=5))
    ax.text(x, y, txt, ha="center", va="center", fontsize=9.5, zorder=6)


def gfac(ax, x, y, fc, s=0.14):
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, fc=fc, ec="none",
                           zorder=5))


def gedge(ax, x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color="0.45", lw=1.0, zorder=2,
            solid_capstyle="round")


def timearrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=9, color="0.5", lw=0.9))
    ax.text(x1, y - 0.16, "time", ha="right", va="top", fontsize=7.5,
            color="0.4")


# ================================================================ fig_fgbasic
fig, ax = plt.subplots(figsize=(COL_W, 1.72))
ax.set_xlim(-0.15, 7.1); ax.set_ylim(-0.75, 1.9); ax.axis("off")
XS = [1.15, 2.75, 4.35, 5.95]
YV, YM = 1.05, 0.12
for i, x in enumerate(XS):
    gvar(ax, x, YV, rf"$\boldsymbol{{\chi}}_{{{i}}}$", FC_VAR_X)
    gedge(ax, x, YV - 0.30, x, YM)
    gfac(ax, x, YM, FC_MEAS)
gedge(ax, 0.30, YV, XS[0] - 0.30, YV)
gfac(ax, 0.30, YV, FC_PRIOR)
for a, b in zip(XS[:-1], XS[1:]):
    gedge(ax, a + 0.30, YV, b - 0.30, YV)
    gfac(ax, (a + b) / 2, YV, FC_PROC)
gedge(ax, XS[-1] + 0.30, YV, XS[-1] + 0.78, YV)
ax.text(XS[-1] + 0.92, YV, r"$\cdots$", fontsize=12, va="center")
# legend row
ly = -0.52
for x0, fc, lab in ((0.55, FC_PRIOR, "prior"), (2.05, FC_PROC, "process"),
                    (3.85, FC_MEAS, "measurement")):
    gfac(ax, x0, ly, fc)
    ax.text(x0 + 0.16, ly, lab, va="center", fontsize=7.8)
timearrow(ax, 5.55, 6.9, ly)
fig.tight_layout(pad=0.15)
fig.savefig(os.path.join(OUT, "fig_fgbasic.pdf"))
fig.savefig(os.path.join(OUT, "fig_fgbasic.png"), dpi=170)
plt.close(fig)
print("fig_fgbasic")

# ================================================================== fig_graph
fig, ax = plt.subplots(figsize=(7.16, 2.55))
ax.set_xlim(-0.2, 15.4); ax.set_ylim(-0.85, 3.35); ax.axis("off")
XS = [2.3, 5.4, 8.5, 11.6]
YX, YB = 2.55, 0.35
YZ = (YX + YB) / 2
for i, x in enumerate(XS):
    gvar(ax, x, YX, rf"$\mathbf{{x}}_{{{i}}}$", FC_VAR_X)
    gvar(ax, x, YB, rf"$\boldsymbol{{\beta}}_{{{i}}}$", FC_VAR_B)
    gedge(ax, x, YX - 0.30, x, YZ + 0.07)
    gedge(ax, x, YB + 0.30, x, YZ - 0.07)
    gfac(ax, x, YZ, FC_MEAS)
for a, b in zip(XS[:-1], XS[1:]):
    for y, fc in ((YX, FC_PROC), (YB, FC_PROC)):
        gedge(ax, a + 0.30, y, b - 0.30, y)
        gfac(ax, (a + b) / 2, y, fc)
# joint prior touches both chains
gfac(ax, 0.75, YZ, FC_PRIOR)
gedge(ax, 0.75, YZ + 0.07, XS[0] - 0.24, YX - 0.18)
gedge(ax, 0.75, YZ - 0.07, XS[0] - 0.24, YB + 0.18)
gedge(ax, XS[-1] + 0.30, YX, XS[-1] + 0.95, YX)
gedge(ax, XS[-1] + 0.30, YB, XS[-1] + 0.95, YB)
ax.text(XS[-1] + 1.12, YX, r"$\cdots$", fontsize=12, va="center")
ax.text(XS[-1] + 1.12, YB, r"$\cdots$", fontsize=12, va="center")
# factor annotations
ax.text((XS[0] + XS[1]) / 2, YX + 0.34,
        r"$\bar{\phi}^{\mathrm{ins}}_k$", fontsize=9, ha="center")
ax.text((XS[0] + XS[1]) / 2, YB - 0.40,
        r"$\bar{\phi}^{\beta}_k$", fontsize=9, ha="center")
ax.text(XS[1] + 0.42, YZ, r"$\bar{\phi}^{\mathrm{map}}_k$", fontsize=9,
        va="center")
ax.text(0.75, YZ - 0.42, r"$\bar{\phi}^{0}$", fontsize=9, ha="center")
# side labels
ax.text(14.0, YX + 0.62, "inertial error chain", fontsize=8, ha="right",
        color="0.35")
ax.text(14.0, YB - 0.62, "compensation chain", fontsize=8, ha="right",
        color="0.35")
timearrow(ax, 12.6, 14.6, YZ)
fig.tight_layout(pad=0.15)
fig.savefig(os.path.join(OUT, "fig_graph.pdf"))
fig.savefig(os.path.join(OUT, "fig_graph.png"), dpi=170)
plt.close(fig)
print("fig_graph")

# ================================================================== fig_track
fig, ax = plt.subplots(figsize=(COL_W, 2.9))
im = map_panel(ax)
ax.plot(tdeg(tlon), tdeg(tlat), color="k", lw=1.3, label="truth")
ax.plot(tdeg(io1 + est[:, 1]), tdeg(il1 + est[:, 0]), color=C_PROPOSED,
        lw=1.0, ls=(0, (4, 2)), label="committed (24.2 m)")
ax.plot(tdeg(tlon[0]), tdeg(tlat[0]), marker="^", color="k", ms=6)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("anomaly [nT]", fontsize=7); cb.ax.tick_params(labelsize=6.5)
ax.legend(loc="upper left", fontsize=6.8, framealpha=0.9)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "fig_track.pdf"))
fig.savefig(os.path.join(OUT, "fig_track.png"), dpi=170)
plt.close(fig)
print("fig_track")

# =================================================================== fig_comp
iTL = slice(17, 36)
tk = idx * dt / 60.0
intf_ref = meas4[idx] - bilin(tl1, tn1)         # interference + noise, truth pos
comp_est = np.einsum("ij,ij->i", A[idx], est[:, iTL]) + est[:, 36]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 2.9), sharex=True,
                             height_ratios=[2.1, 1.0])
a1.plot(tk, intf_ref, color="0.3", lw=0.9, label="reference  $z-h$ at truth")
a1.plot(tk, comp_est, color=C_PROPOSED, lw=0.9, alpha=0.9,
        label=r"estimated  $\mathbf{A}^{\top}\boldsymbol{\beta}+S$")
a1.set_ylabel("[nT]")
a1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=2,
          fontsize=6.8, frameon=False, columnspacing=1.2)
a1.grid(True)
despine(a1)
a2.plot(tk, intf_ref - comp_est, color=C_BASE1, lw=0.7)
a2.axhline(0, color="0.6", lw=0.5)
a2.set_ylabel("residual [nT]", fontsize=8)
a2.set_ylim(-120, 120)
a2.set_xlabel("time [min]")
a2.set_xlim(0, tk[-1])
a2.grid(True)
despine(a2)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "fig_comp.pdf"))
fig.savefig(os.path.join(OUT, "fig_comp.png"), dpi=170)
plt.close(fig)
print("fig_comp")

# ============================================================= fig_breadthbar
ROWS = [("1007.06  M4", 46.7, 48.8, 42.7, 24.2),
        ("1007.06  M5", 17.8, 18.3, 16.0, 11.5),
        ("1007.02  M4", None, 130.0, 74.3, 28.4),
        ("1007.02  M5", 31.6, 30.4, 33.1, 15.8),
        ("1003.02  M4", None, 101.0, 58.2, 19.6),
        ("1003.02  M5", 28.1, 29.5, 21.9, 12.0),
        ("1003.08  M4", "err", 46.6, 48.9, 25.0),
        ("1003.08  M5", 21.1, 20.7, 19.2, 10.5)]
C_WEAK = "#9aa4b2"
SERIES = ((C_WEAK, "EKF, online TL"), (C_BASE1, "EKF+TL+NN"),
          (C_BASE3, "proposed, causal"), (C_PROPOSED, "proposed, committed"))
fig, ax = plt.subplots(figsize=(COL_W, 3.4))
y = np.arange(len(ROWS))[::-1]
h = 0.19
DIVX = 400.0
for k, (color, lab) in enumerate(SERIES):
    for i, r in enumerate(ROWS):
        v = r[k + 1]
        yy = y[i] + (1.5 - k) * h
        if v is None or v == "err":
            ax.barh(yy, DIVX, height=h, color=color, alpha=0.30, hatch="///",
                    edgecolor=color, linewidth=0.4)
            ax.text(DIVX, yy, "  div." if v is None else "  err.",
                    va="center", fontsize=5.6, color="0.45", style="italic")
        else:
            ax.barh(yy, v, height=h, color=color, edgecolor="white",
                    linewidth=0.4)
    ax.barh(np.nan, np.nan, color=color, label=lab)
ax.set_xscale("log")
ax.set_xlim(8, 700)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=7)
ax.set_xlabel("DRMS [m]  (log scale)")
ax.grid(True, axis="x", which="both")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
          fontsize=6.6, frameon=False, columnspacing=1.0, handlelength=1.3)
despine(ax)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "fig_breadthbar.pdf"))
fig.savefig(os.path.join(OUT, "fig_breadthbar.png"), dpi=170)
plt.close(fig)
print("fig_breadthbar")

# =============================================================== fig_lagsweep
LAGS = [30, 60, 150, 300, 600]
SWEEP = {  # (line, mag): ([committed per lag], causal)
    ("1007.06", 4): ([40.2, 37.1, 29.7, 24.2, 21.9], 42.7),
    ("1007.02", 4): ([70.2, 60.4, 43.2, 28.4, 25.8], 74.3),
    ("1003.02", 4): ([48.9, 43.3, 28.8, 19.6, 18.2], 58.2),
    ("1003.08", 4): ([50.3, 41.5, 31.5, 25.0, 24.6], 48.9),
    ("1007.06", 5): ([15.1, 14.3, 12.6, 11.5, 11.4], 16.0),
    ("1007.02", 5): ([27.9, 24.6, 18.3, 15.8, 14.2], 33.1),
    ("1003.02", 5): ([20.1, 18.3, 14.9, 12.0, 10.6], 21.9),
    ("1003.08", 5): ([17.4, 15.8, 12.5, 10.5, 10.7], 19.2),
}
LINESTY = {"1007.06": ("o", C_PROPOSED), "1007.02": ("s", C_BASE1),
           "1003.02": ("^", C_BASE2), "1003.08": ("d", C_BASE3)}
fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.3), sharex=True)
for ax, mag in zip(axes, (4, 5)):
    for line, (mk, c) in LINESTY.items():
        com, cau = SWEEP[(line, mag)]
        ax.plot(LAGS, com, marker=mk, ms=3.2, lw=1.1, color=c, label=line)
        ax.plot(LAGS[-1] * 1.45, cau, marker=mk, ms=3.2, color=c, mfc="white")
    ax.axvline(300, color="0.6", lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([30, 60, 150, 300, 600])
    ax.set_xticklabels(["30", "60", "150", "300", "600"], fontsize=6.6)
    ax.set_xlabel("lag $L$ [s]")
    ax.set_title(f"Mag {mag}", fontsize=8.5)
    ax.grid(True, which="both")
    despine(ax)
axes[0].set_ylabel("committed DRMS [m]")
for ax in axes:
    ax.set_xlim(26, LAGS[-1] * 1.9)
axes[1].text(LAGS[-1] * 1.45, 30.5, "causal", fontsize=6, ha="center",
             color="0.35")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.4,
           frameon=False, bbox_to_anchor=(0.5, -0.045), columnspacing=1.0,
           handlelength=1.3)
fig.subplots_adjust(bottom=0.40)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "fig_lagsweep.pdf"))
fig.savefig(os.path.join(OUT, "fig_lagsweep.png"), dpi=170)
plt.close(fig)
print("fig_lagsweep")
