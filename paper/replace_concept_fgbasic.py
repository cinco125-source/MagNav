#!/usr/bin/env python3
"""Rebuild fig_concept as a pure concept illustration, fig_fgbasic as the
graph the smoother actually maintains, and split the old data panel into its
own fig_signal for Section II.

Run once: python paper/replace_concept_fgbasic.py  (edits make_final_figures.py)
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "make_final_figures.py")

CONCEPT = '''# ================================================================ fig_concept
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
ax.text(cx, 5.32, "uncalibrated aircraft field\\n(cold start: no calibration flight)",
        ha="center", fontsize=7.6, color="#7d5264")
ax.annotate("", xy=(cx + 1.1, cz - 1.35), xytext=(cx + 0.35, cz - 0.42),
            arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.0))
ax.text(cx + 1.32, cz - 1.02, "scalar magnetometer\\n$z=h(\\\\mathbf{p})"
        "+\\\\mathbf{A}^{\\\\top}\\\\boldsymbol{\\\\beta}+S+\\\\eta$",
        fontsize=7.8, va="center")

# --- anomaly map strip (a real grid row) ---------------------------------
row = anom[anom.shape[0] // 2]
strip = np.tile(row, (14, 1))
ax.imshow(strip, extent=[0.4, 6.4, 0.35, 1.25], cmap=CMAP,
          vmin=-np.percentile(np.abs(anom), 99),
          vmax=np.percentile(np.abs(anom), 99), zorder=2,
          interpolation="bilinear")
ax.add_patch(Rectangle((0.4, 0.35), 6.0, 0.9, fill=False, ec="0.4", lw=0.7,
                       zorder=3))
ax.text(3.4, 0.06, "crustal anomaly map  $h(\\\\mathbf{p})$", ha="center",
        fontsize=7.8)
ax.annotate("", xy=(cx, 1.30), xytext=(cx, cz - 1.55),
            arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.0,
                            linestyle=(0, (4, 3))))
ax.text(cx - 0.15, 1.85, "where am I?", fontsize=7.4, style="italic",
        color="0.35", ha="right")

# --- the two coupled unknowns -------------------------------------------
ax.text(7.05, 4.55, "position\\n$\\\\mathbf{p}$", ha="center", fontsize=8,
        color="#1c4587")
ax.text(7.05, 1.55, "compensation\\n$\\\\boldsymbol{\\\\beta}$", ha="center",
        fontsize=8, color="#8a5a00")
ax.annotate("", xy=(7.05, 2.45), xytext=(7.05, 3.75),
            arrowprops=dict(arrowstyle="<|-|>", color="0.4", lw=1.0))
ax.text(7.42, 3.1, "coupled by\\none scalar", fontsize=6.8, color="0.4",
        va="center")

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
ax.annotate("", xy=(bx0 - 0.06, 3.0), xytext=(7.75, 3.0),
            arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.1))

# --- outputs -------------------------------------------------------------
ax.annotate("", xy=(14.05, 3.85), xytext=(bx1 + 0.10, 3.85),
            arrowprops=dict(arrowstyle="-|>", color=C_BASE3, lw=1.3))
ax.text(14.25, 3.85, "causal\\nnow, no look-ahead", fontsize=7.4, va="center",
        color="#7a5c00")
ax.annotate("", xy=(14.05, 2.15), xytext=(bx1 + 0.10, 2.15),
            arrowprops=dict(arrowstyle="-|>", color=C_PROPOSED, lw=1.3))
ax.text(14.25, 2.15, "committed\\nrevised by lag $L$", fontsize=7.4,
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
         label="aircraft field  $z-h(\\\\mathbf{p})$")
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

'''

FGBASIC = '''# ================================================================ fig_fgbasic
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
ax.text(1.72, 2.06, "marginal prior\\n(states older than $L$)", fontsize=7,
        ha="center", color="0.35")
gedge(ax, 1.85, YV, XS[0] - 0.28, YV)
for i, x in enumerate(XS):
    lab = rf"$\\boldsymbol{{\\chi}}_{{k-{len(XS)-1-i}}}$" if i < len(XS)-1 \\
        else r"$\\boldsymbol{\\chi}_{k}$"
    gvar(ax, x, YV, lab, FC_VAR_X, r=0.34)
    gedge(ax, x, YV - 0.34, x, YM + 0.07)
    gfac(ax, x, YM, FC_MEAS)
for a, b in zip(XS[:-1], XS[1:]):
    gedge(ax, a + 0.34, YV, b - 0.34, YV)
    gfac(ax, (a + b) / 2, YV, FC_PROC)
ax.text((XS[0] + XS[1]) / 2, YV + 0.42,
        r"$\\bar{\\phi}^{\\mathrm{ins}}_k$: compounded"
        "\\n$\\\\boldsymbol{\\\\Phi}\\'\\,,\\\\mathbf{Q}\\'$ over $K$ samples",
        fontsize=6.8, ha="center", color="0.35")
ax.text(XS[2], YM - 0.42,
        r"robust map factor  "
        r"$\\rho(\\lVert h+\\mathbf{A}^{\\top}\\boldsymbol{\\beta}+S-z"
        r"\\rVert^{2}_{R})$",
        fontsize=6.8, ha="center", color="0.35")
# causal read-out at the newest state
ax.add_patch(Circle((XS[-1], YV), 0.40, fc="none", ec=C_BASE3, lw=1.4,
                    zorder=6))
ax.annotate("causal output", xy=(XS[-1] + 0.36, YV + 0.22),
            xytext=(13.6, 2.15), fontsize=7.2, color="#7a5c00",
            arrowprops=dict(arrowstyle="-|>", color="#b58900", lw=0.9))
# committed read-out where states leave the lag
ax.annotate("committed output\\n(leaves the lag)", xy=(XS[0] - 0.1, YV - 0.32),
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

'''


def main():
    s = io.open(SRC, encoding="utf-8").read()
    a = s.index("# ================================================================ fig_concept")
    b = s.index("# =============================================================== graph helpers")
    s = s[:a] + CONCEPT + s[b:]
    # fgbasic must come after the helpers it uses
    a = s.index("# ================================================================ fig_fgbasic")
    b = s.index("# ================================================================== fig_graph")
    s = s[:a] + FGBASIC + s[b:]
    if "FancyBboxPatch" not in s.split("\n")[24]:
        s = s.replace(
            "from matplotlib.patches import Circle, Rectangle, FancyArrowPatch",
            "from matplotlib.patches import (Circle, Rectangle, FancyArrowPatch,\n"
            "                                FancyBboxPatch)")
    io.open(SRC, "w", encoding="utf-8").write(s)
    print("concept, signal and fgbasic blocks replaced")


if __name__ == "__main__":
    main()
