#!/usr/bin/env python3
"""Second concept-figure revision: outputs are position and compensation, and
the ground is the real 2-D anomaly map drawn as a sheared plane.

Run once: python paper/replace_concept2.py  (edits make_final_figures.py)
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "make_final_figures.py")

NEW = '''# ================================================================ fig_concept
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
mx0, mx1, my0, my1 = 0.55, 6.05, 0.12, 1.86
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
        "crustal anomaly map  $h(\\\\mathbf{p})$  (Renfrew survey)",
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
ax.text(cx - 0.2, 5.48,
        "uncalibrated aircraft field  (cold start: no calibration flight)",
        ha="center", fontsize=7.6, color="#7d5264")
ax.annotate("", xy=(2.05, 1.62), xytext=(cx - 0.42, cz - 0.55),
            arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.0,
                            linestyle=(0, (4, 3))))
ax.text(1.55, 2.62, "where am I?", fontsize=7.2, style="italic",
        color="0.35", ha="center")

# --- measurement into the estimator --------------------------------------
eqx = 6.35
ax.text(eqx, 3.72, "one scalar reading",
        fontsize=7.2, ha="center", color="0.25")
ax.text(eqx, 3.22,
        "$z=h(\\\\mathbf{p})+\\\\mathbf{A}^{\\\\top}\\\\boldsymbol{\\\\beta}+S$",
        fontsize=8.2, ha="center")
ax.text(eqx, 2.68, "couples position and compensation",
        fontsize=6.6, ha="center", color="0.4")
ax.annotate("", xy=(4.55, 3.35), xytext=(5.1, 3.22),
            arrowprops=dict(arrowstyle="<|-", color="0.35", lw=1.0))
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
ax.text(gxs[0] - 0.34, gy1, "$\\\\mathbf{x}$", fontsize=7.5, ha="right",
        va="center", color="#1c4587")
ax.text(gxs[0] - 0.34, gy2, "$\\\\boldsymbol{\\\\beta}$", fontsize=7.5,
        ha="right", va="center", color="#8a5a00")

# --- outputs: position and compensation ----------------------------------
ax.annotate("", xy=(13.55, 3.95), xytext=(bx1 + 0.12, 3.95),
            arrowprops=dict(arrowstyle="-|>", color="#1c4587", lw=1.4))
ax.text(13.75, 4.42, "position  $\\\\hat{\\\\mathbf{p}}$", fontsize=8.4,
        va="center", color="#1c4587")
ax.text(13.75, 3.92, "causal: now\\ncommitted: after lag $L$", fontsize=6.8,
        va="center", color="0.35")
ax.annotate("", xy=(13.55, 1.95), xytext=(bx1 + 0.12, 1.95),
            arrowprops=dict(arrowstyle="-|>", color="#8a5a00", lw=1.4))
ax.text(13.75, 2.42, "compensation  $\\\\hat{\\\\boldsymbol{\\\\beta}}$",
        fontsize=8.4, va="center", color="#8a5a00")
ax.text(13.75, 1.92, "learned in flight,\\nno calibration flight",
        fontsize=6.8, va="center", color="0.35")
fig.tight_layout(pad=0.25)
fig.savefig(os.path.join(OUT, "fig_concept.pdf"))
fig.savefig(os.path.join(OUT, "fig_concept.png"), dpi=170)
plt.close(fig)
print("fig_concept")

'''


def main():
    s = io.open(SRC, encoding="utf-8").read()
    a = s.index("# ================================================================ fig_concept")
    b = s.index("# ================================================================= fig_signal")
    s = s[:a] + NEW + s[b:]
    io.open(SRC, "w", encoding="utf-8").write(s)
    print("concept block replaced")


if __name__ == "__main__":
    main()
