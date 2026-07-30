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
# Conceptual overview. The ground plane is the real Renfrew anomaly grid,
# sheared to read as terrain under the aircraft; the interference streamlines
# are a computed 2-D dipole field; the joint graph outputs the position and the
# compensation together.
from matplotlib.transforms import Affine2D

fig, ax = plt.subplots(figsize=(7.16, 2.9))
ax.set_xlim(0, 16.4); ax.set_ylim(-0.35, 5.75); ax.axis("off")

# --- real anomaly map as a sheared ground plane --------------------------
i0 = np.searchsorted(glat, bbox["lat"][0]); i1 = np.searchsorted(glat, bbox["lat"][1])
j0 = np.searchsorted(glon, bbox["lon"][0]); j1 = np.searchsorted(glon, bbox["lon"][1])
patch = anom[i0:i1, j0:j1]
mx0, mx1, my0, my1 = 1.45, 6.95, 0.12, 1.86
shear = Affine2D().skew_deg(-32, 0)
imm = ax.imshow(patch, origin="lower", cmap=CMAP,
                extent=[mx0, mx1, my0, my1],
                vmin=-np.percentile(np.abs(patch), 99),
                vmax=np.percentile(np.abs(patch), 99),
                interpolation="bilinear", zorder=2, clip_on=False)
imm.set_transform(shear + ax.transData)
corners = np.array([[mx0, my0], [mx1, my0], [mx1, my1], [mx0, my1]])
sheared = shear.transform(corners)
ax.add_patch(plt.Polygon(sheared, closed=True, fill=False, ec="0.4", lw=0.8,
                         zorder=3))
ax.text(np.mean(sheared[:2, 0]), -0.28,
        "crustal anomaly map  $h(\\mathbf{p})$  (Renfrew survey)",
        ha="center", fontsize=7.6)

# --- aircraft + dipole interference field --------------------------------
cx, cz = 3.15, 4.05
yy, zz = np.meshgrid(np.linspace(-2.1, 2.1, 150), np.linspace(-1.55, 1.55, 150))
rr2 = yy**2 + zz**2 + 1e-3
Bz = (3*zz*yy)/rr2**2.5
By = (2*zz**2 - yy**2)/rr2**2.5
sp = ax.streamplot(yy + cx, zz + cz, Bz, By, density=0.60, linewidth=0.55,
                   color="#b48a9c", arrowsize=0.0, zorder=3)
sp.lines.set_alpha(0.6)
body = np.array([[-1.05, 0], [-0.35, 0.13], [0.55, 0.13], [0.95, 0.02],
                 [0.95, -0.02], [0.55, -0.13], [-0.35, -0.13]])
wing = np.array([[-0.15, 0.06], [0.18, 0.06], [0.05, 0.95], [-0.22, 0.95]])
tail = np.array([[-1.02, 0.04], [-0.78, 0.04], [-0.86, 0.52], [-1.05, 0.52]])
for poly in (body, wing, wing * [1, -1], tail):
    ax.add_patch(plt.Polygon(poly * 1.1 + [cx, cz], closed=True, fc="#37465a",
                             ec="none", zorder=5))
ax.text(cx + 0.55, 5.48,
        "uncalibrated aircraft field  (cold start: no calibration flight)",
        ha="center", fontsize=7.6, color="#7d5264")
ax.annotate("", xy=(2.05, 1.62), xytext=(cx - 0.42, cz - 0.55),
            arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.0,
                            linestyle=(0, (4, 3))))
ax.text(1.55, 2.62, "where am I?", fontsize=7.2, style="italic",
        color="0.35", ha="center")

# --- measurement into the estimator --------------------------------------
eqx = 6.05
ax.text(eqx, 3.72, "one scalar reading",
        fontsize=7.2, ha="center", color="0.25")
ax.text(eqx, 3.22,
        "$z=h(\\mathbf{p})+\\mathbf{A}^{\\top}\\boldsymbol{\\beta}+S$",
        fontsize=8.2, ha="center")
ax.text(eqx, 2.68, "couples position\nand compensation",
        fontsize=6.6, ha="center", color="0.4")
ax.annotate("", xy=(8.30, 3.05), xytext=(7.62, 3.05),
            arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.2))

# --- estimator box --------------------------------------------------------
bx0, bx1, by0, by1 = 8.45, 12.15, 1.05, 5.05
ax.add_patch(FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0,
                            boxstyle="round,pad=0.12", fc="#eef2f8",
                            ec="#37465a", lw=1.1, zorder=3))
ax.text((bx0 + bx1) / 2, by1 - 0.44, "joint factor graph", ha="center",
        fontsize=8.8, color="#22303f")
ax.text((bx0 + bx1) / 2, by1 - 0.92, "iSAM2, fixed lag $L$", ha="center",
        fontsize=7.4, color="0.35")
gy1, gy2 = 2.55, 1.70
gxs = np.linspace(bx0 + 0.72, bx1 - 0.72, 4)
for gx in gxs:
    ax.add_patch(Circle((gx, gy1), 0.16, fc="#dbe7f5", ec="#37465a", lw=0.8,
                        zorder=5))
    ax.add_patch(Circle((gx, gy2), 0.16, fc="#fdeecd", ec="#37465a", lw=0.8,
                        zorder=5))
    ax.plot([gx, gx], [gy2 + 0.16, gy1 - 0.16], color="0.45", lw=0.7, zorder=4)
    ax.add_patch(Rectangle((gx - 0.048, (gy1 + gy2) / 2 - 0.048), 0.096, 0.096,
                           fc=C_PROPOSED, ec="none", zorder=6))
for gy in (gy1, gy2):
    for a, b in zip(gxs[:-1], gxs[1:]):
        ax.plot([a + 0.16, b - 0.16], [gy, gy], color="0.45", lw=0.7, zorder=4)
ax.text(gxs[0] - 0.34, gy1, "$\\mathbf{x}$", fontsize=7.5, ha="right",
        va="center", color="#1c4587")
ax.text(gxs[0] - 0.34, gy2, "$\\boldsymbol{\\beta}$", fontsize=7.5,
        ha="right", va="center", color="#8a5a00")

# --- outputs: position and compensation ----------------------------------
ax.annotate("", xy=(13.55, 3.95), xytext=(bx1 + 0.12, 3.95),
            arrowprops=dict(arrowstyle="-|>", color="#1c4587", lw=1.4))
ax.text(13.75, 4.42, "position  $\\hat{\\mathbf{p}}$", fontsize=8.4,
        va="center", color="#1c4587")
ax.text(13.75, 3.92, "causal: now\ncommitted: after lag $L$", fontsize=6.8,
        va="center", color="0.35")
ax.annotate("", xy=(13.55, 1.95), xytext=(bx1 + 0.12, 1.95),
            arrowprops=dict(arrowstyle="-|>", color="#8a5a00", lw=1.4))
ax.text(13.75, 2.42, "compensation  $\\hat{\\boldsymbol{\\beta}}$",
        fontsize=8.4, va="center", color="#8a5a00")
ax.text(13.75, 1.92, "learned in flight,\nno calibration flight",
        fontsize=6.8, va="center", color="0.35")
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


# ================================================================== fig_graph
# Our factor graph, drawn with the conventions of the group's terrain-graph
# figures: colored edges per factor type, the real anomaly map as the ground,
# a shaded sliding-lag band, and the marginal prior at its boundary.
C_INS = "#c0392b"      # inertial chain factors and edges
C_BET = "#1c4587"      # compensation chain factors and edges
C_MAP = "#2e8b57"      # robust map factors
C_PRI = C_ACCENT       # joint marginal prior
YX, YB, YZ, YMAP = 4.05, 2.55, 3.30, 1.62

fig, ax = plt.subplots(figsize=(7.16, 3.35))
ax.set_xlim(-0.2, 16.6); ax.set_ylim(-0.30, 5.35); ax.axis("off")
XS = [3.1, 5.7, 8.3, 10.9, 13.5]

# sliding lag band behind both chains (arrow band, oldest to newest)
band = FancyBboxPatch((XS[0] - 0.95, YB - 0.62), XS[-1] - XS[0] + 1.9,
                      YX - YB + 1.24, boxstyle="round,pad=0.05",
                      fc="0.93", ec="none", zorder=0)
ax.add_patch(band)
ax.annotate("", xy=(16.05, YZ), xytext=(15.15, YZ),
            arrowprops=dict(arrowstyle="-|>", color="0.75", lw=5,
                            mutation_scale=16), zorder=0)
ax.text(XS[0] - 0.72, YX + 0.44, "sliding window, lag $L$", fontsize=7.4,
        ha="left", color="0.45", zorder=1)

def vnode(x, y, txt, fc):
    ax.add_patch(FancyBboxPatch((x - 0.42, y - 0.30), 0.84, 0.60,
                                boxstyle="round,pad=0.06", fc=fc,
                                ec="#22303f", lw=1.0, zorder=6))
    ax.text(x, y, txt, ha="center", va="center", fontsize=8.6, zorder=7)

def sq(x, y, color, s=0.17, z=5):
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, fc=color,
                           ec="white", lw=0.5, zorder=z))

def cedge(x1, y1, x2, y2, color, lw=1.2, ls="-", z=2):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, zorder=z,
            solid_capstyle="round")

# marginalized past: ghost node + joint marginal prior square
ax.add_patch(Circle((0.75, YZ), 0.30, fc="0.92", ec="0.6", lw=0.9,
                    ls=(0, (2, 2)), zorder=4))
ax.text(0.75, YZ, "$\\cdots$", ha="center", va="center", fontsize=8,
        color="0.5", zorder=5)
sq(1.75, YZ, C_PRI, s=0.20)
cedge(1.05, YZ, 1.65, YZ, "0.6", lw=1.0, ls=(0, (2, 2)))
cedge(1.85, YZ + 0.05, XS[0] - 0.42, YX - 0.10, C_PRI, lw=1.1)
cedge(1.85, YZ - 0.05, XS[0] - 0.42, YB + 0.10, C_PRI, lw=1.1)
ax.text(1.72, YZ - 0.55, "marginal\nprior", fontsize=6.8, ha="center",
        color="0.35")

# chains
labs_x = ["$\\mathbf{x}_{k-L}$", "$\\mathbf{x}$", "$\\cdots$",
          "$\\mathbf{x}$", "$\\mathbf{x}_{k}$"]
labs_b = ["$\\boldsymbol{\\beta}_{k-L}$", "$\\boldsymbol{\\beta}$",
          "$\\cdots$", "$\\boldsymbol{\\beta}$",
          "$\\boldsymbol{\\beta}_{k}$"]
for a, b in zip(XS[:-1], XS[1:]):
    cedge(a, YX, b, YX, C_INS, lw=1.3)
    sq((a + b) / 2, YX, C_INS)
    cedge(a, YB, b, YB, C_BET, lw=1.3)
    sq((a + b) / 2, YB, C_BET)
for i, x in enumerate(XS):
    # map factor connects BOTH chains: the joint measurement
    cedge(x, YZ + 0.0, x, YX - 0.10, C_MAP, lw=1.0)
    cedge(x, YB + 0.10, x, YZ, C_MAP, lw=1.0)
    cedge(x, YMAP + 0.09, x, YB - 0.10, C_MAP, lw=1.0)
    sq(x, YMAP, C_MAP, s=0.19)
    vnode(x, YX, labs_x[i], "#f6d9d3")
    vnode(x, YB, labs_b[i], "#d8e4f5")
    # interpolation fan into the real map
    for dxf in (-0.42, 0.0, 0.42):
        cedge(x, YMAP - 0.10, x + dxf, 1.02, "#e2c34c", lw=0.8,
              ls=(0, (2, 1.6)), z=3)
        ax.plot(x + dxf, 1.02, marker="o", ms=2.2, color="#e2c34c", zorder=4)

# factor symbols
ax.text((XS[2] + XS[3]) / 2, YX + 0.42, "$\\bar{\\phi}^{\\mathrm{ins}}_k$",
        fontsize=8.2, ha="center", color=C_INS)
ax.text((XS[0] + XS[1]) / 2, YB - 0.46, "$\\bar{\\phi}^{\\beta}_k$",
        fontsize=8.2, ha="center", color=C_BET)
ax.text(XS[1] + 0.52, YZ + 0.04, "$\\bar{\\phi}^{\\mathrm{map}}_k$",
        fontsize=8.2, va="center", color=C_MAP)

# read-out points
ax.add_patch(FancyBboxPatch((XS[-1] - 0.52, YB - 0.40), 1.04, YX - YB + 0.80,
                            boxstyle="round,pad=0.07", fill=False,
                            ec=C_BASE3, lw=1.3, zorder=8))
ax.text(XS[-1] + 0.75, YX + 0.52, "causal\noutput", fontsize=6.9,
        color="#7a5c00", ha="left")
ax.annotate("committed output (leaves the lag)",
            xy=(XS[0] - 0.60, YB - 0.48), xytext=(3.75, 1.16),
            fontsize=6.9, color="#0b5394",
            arrowprops=dict(arrowstyle="-|>", color=C_PROPOSED, lw=0.9))

# the real anomaly map as the ground
i0 = np.searchsorted(glat, bbox["lat"][0]); i1 = np.searchsorted(glat, bbox["lat"][1])
j0 = np.searchsorted(glon, bbox["lon"][0]); j1 = np.searchsorted(glon, bbox["lon"][1])
patch = anom[i0:i1, j0:j1]
strip = patch[patch.shape[0] // 3, None, :].repeat(2, 0)
strip = patch[: patch.shape[0] // 3]
ax.imshow(strip, origin="lower", cmap=CMAP, extent=[1.2, 15.4, 0.02, 1.02],
          vmin=-np.percentile(np.abs(patch), 99),
          vmax=np.percentile(np.abs(patch), 99), interpolation="bilinear",
          zorder=2, aspect="auto")
ax.add_patch(Rectangle((1.2, 0.02), 14.2, 1.0, fill=False, ec="0.35", lw=0.8,
                       zorder=3))
ax.text(8.3, -0.26, "anomaly map $h(\\cdot)$ (Renfrew survey)", ha="center",
        fontsize=7.6)

# legend, factor types
lx, ly = 0.25, 5.05
for dx, c, lab in ((0.0, C_PRI, "marginal prior"),
                   (2.7, C_INS, "inertial $\\bar{\\phi}^{\\mathrm{ins}}$"),
                   (5.6, C_BET, "compensation walk $\\bar{\\phi}^{\\beta}$"),
                   (9.6, C_MAP, "robust map $\\bar{\\phi}^{\\mathrm{map}}$")):
    sq(lx + dx, ly, c, s=0.16, z=9)
    ax.text(lx + dx + 0.18, ly, lab, fontsize=7.0, va="center")
fig.tight_layout(pad=0.2)
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

# (b) zoomed latitude/longitude window where the navigation paths visibly
# separate from the truth: centred where the INS deviation is largest.
_w10 = int(600 / dt)                     # search after the warm-up only
i_max = _w10 + int(np.argmax(np.hypot(e_ekf_e, e_ekf_n)[_w10:]))
clat0, clon0 = tdeg(tlat[i_max]), tdeg(tlon[i_max])
half_lat = 380.0 / R_EARTH * 180 / np.pi              # ~0.38 km half-height
half_lon = 540.0 / (R_EARTH * np.cos(tlat[i_max])) * 180 / np.pi
zl = dict(lat=(clat0 - half_lat, clat0 + half_lat),
          lon=(clon0 - half_lon, clon0 + half_lon))
axm.add_patch(Rectangle((zl["lon"][0], zl["lat"][0]),
                        2 * half_lon, 2 * half_lat, fill=False, ec="k",
                        lw=1.0, zorder=6))

axe = fig.add_subplot(gs[1])
axe.imshow(anom, origin="lower", cmap=CMAP,
           extent=[glon[0], glon[-1], glat[0], glat[-1]],
           vmin=-np.percentile(np.abs(anom), 99),
           vmax=np.percentile(np.abs(anom), 99), aspect="auto",
           interpolation="bilinear", alpha=0.45)
axe.plot(tdeg(ilon), tdeg(ilat), ":", color="0.3", lw=1.1, label="INS")
axe.plot(tdeg(_get("ekf_lon")), tdeg(_get("ekf_lat")), color="#8b95a5",
         lw=1.0, label="EKF, online TL")
axe.plot(tdeg(_get("nn_lon")), tdeg(_get("nn_lat")), color=C_BASE1, lw=1.0,
         label="EKF+TL+NN")
axe.plot(tdeg(io1 + est_rt[:, 1]), tdeg(il1 + est_rt[:, 0]), color=C_BASE3,
         lw=1.0, label="proposed, causal")
axe.plot(tdeg(io1 + est[:, 1]), tdeg(il1 + est[:, 0]), color=C_PROPOSED,
         lw=1.2, ls=(0, (5, 2)), label="proposed, committed")
axe.plot(tdeg(tlon), tdeg(tlat), color="k", lw=1.5, label="truth")
axe.set_xlim(*zl["lon"]); axe.set_ylim(*zl["lat"])
axe.set_xlabel("longitude [deg]"); axe.set_ylabel("latitude [deg]")
axe.ticklabel_format(useOffset=False)
axe.tick_params(labelsize=6)
axe.legend(loc="upper left", fontsize=6.0, framealpha=0.92,
           handlelength=1.4, labelspacing=0.3)
axe.set_title("(b) zoom: paths separate from the truth", fontsize=8.5)
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
