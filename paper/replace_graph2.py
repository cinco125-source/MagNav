#!/usr/bin/env python3
"""Remake fig_graph in the style of the terrain-graph figures the group uses:
per-factor-type colors with a legend, rounded variable nodes, the real anomaly
map as the ground of the figure with interpolation fans into it, the sliding
lag window as a shaded band, the marginal prior at its boundary, and the
causal/committed read-out points. Also removes the now-redundant fig_fgbasic
block.

Run once: python paper/replace_graph2.py  (edits make_final_figures.py)
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "make_final_figures.py")

NEW = '''# ================================================================== fig_graph
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
ax.text(15.05, YX + 1.02, "sliding window, lag $L$", fontsize=7.6,
        ha="right", color="0.35")

# time ticks
for x, lab in zip(XS, ["$t_{k-L}$", "", "$\\\\cdots$", "", "$t_{k}$"]):
    if lab:
        ax.text(x, YX + 1.02, lab, ha="center", fontsize=7.6, color="0.35")

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
ax.text(0.75, YZ, "$\\\\cdots$", ha="center", va="center", fontsize=8,
        color="0.5", zorder=5)
sq(1.75, YZ, C_PRI, s=0.20)
cedge(1.05, YZ, 1.65, YZ, "0.6", lw=1.0, ls=(0, (2, 2)))
cedge(1.85, YZ + 0.05, XS[0] - 0.42, YX - 0.10, C_PRI, lw=1.1)
cedge(1.85, YZ - 0.05, XS[0] - 0.42, YB + 0.10, C_PRI, lw=1.1)
ax.text(1.72, YZ - 0.55, "marginal\\nprior", fontsize=6.8, ha="center",
        color="0.35")

# chains
labs_x = ["$\\\\mathbf{x}_{k-L}$", "$\\\\mathbf{x}$", "$\\\\cdots$",
          "$\\\\mathbf{x}$", "$\\\\mathbf{x}_{k}$"]
labs_b = ["$\\\\boldsymbol{\\\\beta}_{k-L}$", "$\\\\boldsymbol{\\\\beta}$",
          "$\\\\cdots$", "$\\\\boldsymbol{\\\\beta}$",
          "$\\\\boldsymbol{\\\\beta}_{k}$"]
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
ax.text((XS[0] + XS[1]) / 2, YX + 0.42, "$\\\\bar{\\\\phi}^{\\\\mathrm{ins}}_k$",
        fontsize=8.2, ha="center", color=C_INS)
ax.text((XS[0] + XS[1]) / 2, YB - 0.46, "$\\\\bar{\\\\phi}^{\\\\beta}_k$",
        fontsize=8.2, ha="center", color=C_BET)
ax.text(XS[1] + 0.52, YZ + 0.04, "$\\\\bar{\\\\phi}^{\\\\mathrm{map}}_k$",
        fontsize=8.2, va="center", color=C_MAP)

# read-out points
ax.add_patch(FancyBboxPatch((XS[-1] - 0.52, YB - 0.40), 1.04, YX - YB + 0.80,
                            boxstyle="round,pad=0.07", fill=False,
                            ec=C_BASE3, lw=1.3, zorder=8))
ax.text(XS[-1] + 0.75, YX + 0.52, "causal\\noutput", fontsize=6.9,
        color="#7a5c00", ha="left")
ax.annotate("committed output (leaves the lag)",
            xy=(XS[0] - 0.60, YB - 0.48), xytext=(3.4, 0.16),
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
ax.text(8.3, -0.26, "anomaly map $h(\\\\cdot)$ (Renfrew survey)", ha="center",
        fontsize=7.6)

# legend, factor types
lx, ly = 0.25, 5.05
for dx, c, lab in ((0.0, C_PRI, "marginal prior"),
                   (2.7, C_INS, "inertial $\\\\bar{\\\\phi}^{\\\\mathrm{ins}}$"),
                   (5.6, C_BET, "compensation walk $\\\\bar{\\\\phi}^{\\\\beta}$"),
                   (9.6, C_MAP, "robust map $\\\\bar{\\\\phi}^{\\\\mathrm{map}}$")):
    sq(lx + dx, ly, c, s=0.16, z=9)
    ax.text(lx + dx + 0.18, ly, lab, fontsize=7.0, va="center")
fig.tight_layout(pad=0.2)
fig.savefig(os.path.join(OUT, "fig_graph.pdf"))
fig.savefig(os.path.join(OUT, "fig_graph.png"), dpi=170)
plt.close(fig)
print("fig_graph")

'''


def main():
    s = io.open(SRC, encoding="utf-8").read()
    # drop the fgbasic block entirely
    a = s.index("# ================================================================ fig_fgbasic")
    b = s.index("# ================================================================== fig_graph")
    s = s[:a] + s[b:]
    # replace the graph block
    a = s.index("# ================================================================== fig_graph")
    b = s.index("# ================================================================== fig_track")
    s = s[:a] + NEW + s[b:]
    io.open(SRC, "w", encoding="utf-8").write(s)
    print("fgbasic removed, graph block replaced")


if __name__ == "__main__":
    main()
