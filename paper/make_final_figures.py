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
from matplotlib.patches import (Circle, Rectangle, FancyArrowPatch,
                                FancyBboxPatch)

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
# Conceptual overview, no data axes: an aircraft carrying an uncalibrated
# magnetic signature flies over the anomaly map; one scalar reading mixes the
# two; the joint graph estimates position and compensation together and emits
# a causal and a committed answer. The dipole streamlines are computed from a
# real 2-D dipole field; the map strip is a row of the actual anomaly grid.
fig, ax = plt.subplots(figsize=(7.16, 2.75))
ax.set_xlim(0, 16.4); ax.set_ylim(-0.3, 5.6); ax.axis("off")

# --- aircraft + dipole interference field --------------------------------
cx, cz = 2.9, 3.85
yy, zz = np.meshgrid(np.linspace(-2.3, 2.3, 160), np.linspace(-1.75, 1.75, 160))
rr2 = yy**2 + zz**2 + 1e-3
Bz = (3*zz*yy)/rr2**2.5
By = (2*zz**2 - yy**2)/rr2**2.5
sp = ax.streamplot(yy + cx, zz + cz, Bz, By, density=0.62, linewidth=0.55,
                   color="#b48a9c", arrowsize=0.0, zorder=1)
sp.lines.set_alpha(0.65)
body = np.array([[-1.05, 0], [-0.35, 0.13], [0.55, 0.13], [0.95, 0.02],
                 [0.95, -0.02], [0.55, -0.13], [-0.35, -0.13]])
wing = np.array([[-0.15, 0.06], [0.18, 0.06], [0.05, 0.95], [-0.22, 0.95]])
tail = np.array([[-1.02, 0.04], [-0.78, 0.04], [-0.86, 0.52], [-1.05, 0.52]])
for poly in (body, wing, wing * [1, -1], tail):
    ax.add_patch(plt.Polygon(poly * 1.15 + [cx, cz], closed=True, fc="#37465a",
                             ec="none", zorder=4))
ax.text(cx, 5.32, "uncalibrated aircraft field\n(cold start: no calibration flight)",
        ha="center", fontsize=7.6, color="#7d5264")
ax.annotate("", xy=(cx + 1.1, cz - 1.35), xytext=(cx + 0.35, cz - 0.42),
            arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.0))
ax.text(cx + 1.28, cz - 1.02,
        "scalar magnetometer\n"
        "$z=h(\\mathbf{p})+\\mathbf{A}^{\\top}\\boldsymbol{\\beta}+S$",
        fontsize=7.4, va="center")

# --- anomaly map strip (a real grid row) ---------------------------------
row = anom[anom.shape[0] // 2]
strip = np.tile(row, (14, 1))
ax.imshow(strip, extent=[0.4, 6.4, 0.35, 1.25], cmap=CMAP,
          vmin=-np.percentile(np.abs(anom), 99),
          vmax=np.percentile(np.abs(anom), 99), zorder=2,
          interpolation="bilinear")
ax.add_patch(Rectangle((0.4, 0.35), 6.0, 0.9, fill=False, ec="0.4", lw=0.7,
                       zorder=3))
ax.text(3.4, 0.06, "crustal anomaly map  $h(\\mathbf{p})$", ha="center",
        fontsize=7.8)
ax.annotate("", xy=(cx, 1.30), xytext=(cx, cz - 1.55),
            arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.0,
                            linestyle=(0, (4, 3))))
ax.text(cx - 0.15, 1.85, "where am I?", fontsize=7.4, style="italic",
        color="0.35", ha="right")

# --- the two coupled unknowns -------------------------------------------
ax.text(7.55, 4.55, "position\n$\\mathbf{p}$", ha="center", fontsize=8,
        color="#1c4587")
ax.text(7.55, 1.55, "compensation\n$\\boldsymbol{\\beta}$", ha="center",
        fontsize=8, color="#8a5a00")
ax.annotate("", xy=(7.55, 2.45), xytext=(7.55, 3.75),
            arrowprops=dict(arrowstyle="<|-|>", color="0.4", lw=1.0))
ax.text(7.55, 3.06, "coupled", fontsize=6.6, color="0.4",
        ha="center", va="center",
        bbox=dict(fc="white", ec="none", pad=0.5))

# --- estimator box -------------------------------------------------------
bx0, bx1, by0, by1 = 8.55, 12.45, 1.15, 4.85
ax.add_patch(FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0,
                            boxstyle="round,pad=0.12", fc="#eef2f8",
                            ec="#37465a", lw=1.1, zorder=3))
ax.text((bx0 + bx1) / 2, by1 - 0.42, "joint factor graph", ha="center",
        fontsize=8.6, color="#22303f")
ax.text((bx0 + bx1) / 2, by1 - 0.88, "iSAM2, fixed lag $L$", ha="center",
        fontsize=7.4, color="0.35")
gy1, gy2 = 2.55, 1.72
gxs = np.linspace(bx0 + 0.75, bx1 - 0.75, 4)
for gx in gxs:
    ax.add_patch(Circle((gx, gy1), 0.155, fc="#dbe7f5", ec="#37465a", lw=0.8,
                        zorder=5))
    ax.add_patch(Circle((gx, gy2), 0.155, fc="#fdeecd", ec="#37465a", lw=0.8,
                        zorder=5))
    ax.plot([gx, gx], [gy2 + 0.15, gy1 - 0.15], color="0.45", lw=0.7, zorder=4)
    ax.add_patch(Rectangle((gx - 0.045, (gy1 + gy2) / 2 - 0.045), 0.09, 0.09,
                           fc=C_PROPOSED, ec="none", zorder=6))
for gy in (gy1, gy2):
    for a, b in zip(gxs[:-1], gxs[1:]):
        ax.plot([a + 0.15, b - 0.15], [gy, gy], color="0.45", lw=0.7, zorder=4)
ax.annotate("", xy=(bx0 - 0.06, 3.0), xytext=(8.05, 3.0),
            arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.1))

# --- outputs -------------------------------------------------------------
ax.annotate("", xy=(14.05, 3.85), xytext=(bx1 + 0.10, 3.85),
            arrowprops=dict(arrowstyle="-|>", color=C_BASE3, lw=1.3))
ax.text(14.25, 3.85, "causal\nnow, no look-ahead", fontsize=7.4, va="center",
        color="#7a5c00")
ax.annotate("", xy=(14.05, 2.15), xytext=(bx1 + 0.10, 2.15),
            arrowprops=dict(arrowstyle="-|>", color=C_PROPOSED, lw=1.3))
ax.text(14.25, 2.15, "committed\nrevised by lag $L$", fontsize=7.4,
        va="center", color="#0b5394")
fig.tight_layout(pad=0.25)
fig.savefig(os.path.join(OUT, "fig_concept.pdf"))
fig.savefig(os.path.join(OUT, "fig_concept.png"), dpi=170)
plt.close(fig)
print("fig_concept")

# ================================================================= fig_signal
# The old concept panel (b): the measured interference against the map signal,
# now a stand-alone figure for the problem-formulation section.
fig, axs = plt.subplots(figsize=(COL_W, 2.15))
t_min = np.arange(meas4.size) * dt / 60.0
h_track = bilin(tlat, tlon)
map_sig = h_track - h_track.mean()
intf = meas4 - h_track
axs.plot(t_min, intf, color=C_BASE1, lw=0.7,
         label="aircraft field  $z-h(\\mathbf{p})$")
axs.plot(t_min, map_sig, color=C_PROPOSED, lw=0.8,
         label="map signal along track")
axs.axhline(0, color="0.6", lw=0.5)
axs.set_xlabel("time [min]"); axs.set_ylabel("[nT]")
axs.set_xlim(0, t_min[-1])
axs.legend(loc="center right", fontsize=7, framealpha=0.9)
axs.grid(True)
despine(axs)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "fig_signal.pdf"))
fig.savefig(os.path.join(OUT, "fig_signal.png"), dpi=170)
plt.close(fig)
print("fig_signal")

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
# The graph the smoother actually maintains: joint states at the 1 Hz cadence,
# one compounded process factor per interval, one robust map factor per state,
# the marginal prior standing in for everything older than the lag, and the
# causal / committed read-out points.
fig, ax = plt.subplots(figsize=(7.16, 2.15))
ax.set_xlim(-0.3, 16.2); ax.set_ylim(-1.05, 2.75); ax.axis("off")
XS = [2.6, 5.0, 7.4, 9.8, 12.2]
YV, YM = 1.35, 0.28
# ghost states already marginalized
for gx, al in ((0.2, 0.25), (1.1, 0.4)):
    ax.add_patch(Circle((gx, YV), 0.26, fc="none", ec="0.65", lw=0.9,
                        ls=(0, (2, 2)), alpha=al, zorder=3))
# marginal prior replacing them
gfac(ax, 1.85, YV, FC_PRIOR)
ax.text(0.98, 1.96, "marginal prior\n(states older than $L$)", fontsize=7,
        ha="center", color="0.35")
gedge(ax, 1.85, YV, XS[0] - 0.44, YV)
for i, x in enumerate(XS):
    lab = rf"$\boldsymbol{{\chi}}_{{k-{len(XS)-1-i}}}$" if i < len(XS)-1 \
        else r"$\boldsymbol{\chi}_{k}$"
    gvar(ax, x, YV, lab, FC_VAR_X, r=0.44)
    gedge(ax, x, YV - 0.44, x, YM + 0.07)
    gfac(ax, x, YM, FC_MEAS)
for a, b in zip(XS[:-1], XS[1:]):
    gedge(ax, a + 0.44, YV, b - 0.44, YV)
    gfac(ax, (a + b) / 2, YV, FC_PROC)
ax.text((XS[2] + XS[3]) / 2, YV + 0.50,
        "$\\bar{\\phi}^{\\mathrm{ins}}_k$: compounded\n"
        "$\\boldsymbol{\\Phi}^{\\prime},\\mathbf{Q}^{\\prime}$ over $K$ samples",
        fontsize=6.8, ha="center", color="0.35")
ax.text(XS[2], YM - 0.42,
        r"robust map factor  "
        r"$\rho(\Vert h+\mathbf{A}^{\top}\boldsymbol{\beta}+S-z"
        r"\Vert^{2}_{R})$",
        fontsize=6.8, ha="center", color="0.35")
# causal read-out at the newest state
ax.add_patch(Circle((XS[-1], YV), 0.52, fc="none", ec=C_BASE3, lw=1.4,
                    zorder=6))
ax.annotate("causal output", xy=(XS[-1] + 0.36, YV + 0.22),
            xytext=(13.6, 2.15), fontsize=7.2, color="#7a5c00",
            arrowprops=dict(arrowstyle="-|>", color="#b58900", lw=0.9))
# committed read-out where states leave the lag
ax.annotate("committed output\n(leaves the lag)", xy=(XS[0] - 0.1, YV - 0.32),
            xytext=(2.9, -0.75), fontsize=7.2, color="#0b5394",
            arrowprops=dict(arrowstyle="-|>", color=C_PROPOSED, lw=0.9))
# lag brace
ax.annotate("", xy=(XS[0] - 0.42, 2.35), xytext=(XS[-1] + 0.42, 2.35),
            arrowprops=dict(arrowstyle="<->", color="0.5", lw=0.9))
ax.text((XS[0] + XS[-1]) / 2, 2.52, "lag $L$ (retained states)", ha="center",
        fontsize=7.4, color="0.35")
timearrow(ax, 13.9, 15.8, -0.55)
fig.tight_layout(pad=0.2)
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
# (a) geographic context: on the 60 km map every bounded method overlays the
# truth to line width, so paths are drawn there only for context; (b) the same
# paths in the north/east error plane, metres about the true position, where
# the methods actually separate. Baselines from baseline_tracks (10 Hz),
# proposed from the 1 Hz npz.
import csv as _csv
_rows = list(_csv.DictReader(open(os.path.join(
    HERE, "..", "research", "baseline_tracks_1007_06_m4.csv"))))
_get = lambda k: np.array([float(r[k]) for r in _rows])
_cos = np.cos(tlat)
def _err(lat, lon):
    return ((lon - tlon) * R_EARTH * _cos, (lat - tlat) * R_EARTH)
e_ekf_e, e_ekf_n = _err(_get("ekf_lat"), _get("ekf_lon"))
e_nn_e, e_nn_n = _err(_get("nn_lat"), _get("nn_lon"))
e_ins_e, e_ins_n = _err(ilat, ilon)
_cos1 = np.cos(tl1)
e_com_e = (io1 + est[:, 1] - tn1) * R_EARTH * _cos1
e_com_n = (il1 + est[:, 0] - tl1) * R_EARTH
e_cau_e = (io1 + est_rt[:, 1] - tn1) * R_EARTH * _cos1
e_cau_n = (il1 + est_rt[:, 0] - tl1) * R_EARTH

fig = plt.figure(figsize=(7.16, 3.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.52)
axm = fig.add_subplot(gs[0])
im = map_panel(axm)
axm.plot(tdeg(tlon), tdeg(tlat), color="k", lw=1.2, label="truth")
axm.plot(tdeg(io1 + est[:, 1]), tdeg(il1 + est[:, 0]), color=C_PROPOSED,
         lw=0.9, ls=(0, (4, 2)), label="committed")
axm.plot(tdeg(tlon[0]), tdeg(tlat[0]), marker="^", color="k", ms=6)
cb = fig.colorbar(im, ax=axm, fraction=0.052, pad=0.02)
cb.set_label("anomaly [nT]", fontsize=7, labelpad=1)
cb.ax.tick_params(labelsize=6)
axm.legend(loc="upper left", fontsize=6.6, framealpha=0.9)
axm.set_title("(a) flown line on the map", fontsize=8.5)

axe = fig.add_subplot(gs[1])
w10 = int(600 / dt)                       # warm-up cut, 10 Hz series
w1 = 600                                  # warm-up cut, 1 Hz series
axe.plot(e_ins_e[w10::40], e_ins_n[w10::40], ":", color="0.55", lw=0.9,
         label="INS (drifts out)")
axe.plot(e_ekf_e[w10::20], e_ekf_n[w10::20], color="#9aa4b2", lw=0.65,
         alpha=0.9, label="EKF, online TL")
axe.plot(e_nn_e[w10::20], e_nn_n[w10::20], color=C_BASE1, lw=0.65,
         alpha=0.9, label="EKF+TL+NN")
axe.plot(e_cau_e[w1::2], e_cau_n[w1::2], color=C_BASE3, lw=0.65, alpha=0.9,
         label="proposed, causal")
axe.plot(e_com_e[w1:], e_com_n[w1:], color=C_PROPOSED, lw=1.0,
         label="proposed, committed")
axe.plot(0, 0, marker="+", color="k", ms=8, mew=1.4, zorder=6)
axe.set_xlim(-140, 140); axe.set_ylim(-140, 140)
axe.set_aspect("equal")
axe.set_xlabel("east error [m]"); axe.set_ylabel("north error [m]")
axe.grid(True)
axe.legend(loc="lower right", fontsize=6.0, framealpha=0.9,
           handlelength=1.3, labelspacing=0.3)
axe.set_title("(b) paths about the true position", fontsize=8.5)
despine(axe)
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
