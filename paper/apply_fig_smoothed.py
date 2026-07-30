#!/usr/bin/env python3
"""Figure-side pass for the terminology unification and the two layout fixes:
'committed' becomes 'smoothed' in every legend and annotation, the lag-sweep
legend drops clear of the axis labels, and the factor-graph figure gets a 3-D
relief rendering of the real anomaly map plus an explicit scalar-measurement
annotation on the map factors.

Run: python paper/apply_fig_smoothed.py
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def patch(path, pairs):
    p = os.path.join(HERE, path)
    s = io.open(p, encoding="utf-8").read()
    missed = []
    for old, new in pairs:
        if old in s:
            s = s.replace(old, new)
        else:
            missed.append(old)
    io.open(p, "w", encoding="utf-8").write(s)
    print(f"{path}: {len(pairs)-len(missed)}/{len(pairs)}")
    for m in missed:
        print(f"  NOT FOUND: {' '.join(m.split())[:64]}")


MAIN = [
    # concept output label
    ("causal: now\\ncommitted: after lag $L$", "causal: now\\nsmoothed: after lag $L$"),
    ("causal: now\ncommitted: after lag $L$", "causal: now\nsmoothed: after lag $L$"),
    # track legend
    ("label=\"committed\")", "label=\"smoothed\")"),
    ("label=\"proposed, committed\")", "label=\"proposed, smoothed\")"),
    # graph annotation
    ("ax.annotate(\"committed output (leaves the lag)\",",
     "ax.annotate(\"smoothed output (leaves the lag)\","),
    # breadth bar series
    ("(C_PROPOSED, \"proposed, committed\")", "(C_PROPOSED, \"proposed, smoothed\")"),
    # lag sweep ylabel and legend placement
    ("axes[0].set_ylabel(\"committed DRMS [m]\")",
     "axes[0].set_ylabel(\"smoothed DRMS [m]\")"),
    ("bbox_to_anchor=(0.5, -0.045), columnspacing=1.0,",
     "bbox_to_anchor=(0.5, -0.135), columnspacing=1.0,"),
    ("fig.subplots_adjust(bottom=0.40)", "fig.subplots_adjust(bottom=0.46)"),
    # graph: flat strip becomes a 3-D relief of the same real map
    ("""# the real anomaly map as the ground
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
        fontsize=7.6)""",
     """# the real anomaly map as a 3-D relief ground (height = anomaly strength)
i0 = np.searchsorted(glat, bbox["lat"][0]); i1 = np.searchsorted(glat, bbox["lat"][1])
j0 = np.searchsorted(glon, bbox["lon"][0]); j1 = np.searchsorted(glon, bbox["lon"][1])
patch = anom[i0:i1, j0:j1]
band3 = patch[: patch.shape[0] // 3][::5, ::3]
vmax3 = np.percentile(np.abs(band3), 99)
X3, Y3 = np.meshgrid(np.linspace(0, 1, band3.shape[1]),
                     np.linspace(0, 1, band3.shape[0]))
ax3 = fig.add_axes([0.035, -0.10, 0.93, 0.44], projection="3d")
ax3.plot_surface(X3, Y3, np.clip(band3, -vmax3, vmax3), cmap=CMAP,
                 vmin=-vmax3, vmax=vmax3, rstride=1, cstride=1, linewidth=0,
                 antialiased=True)
ax3.set_box_aspect((10.5, 2.2, 1.15))
ax3.view_init(elev=42, azim=-90)
ax3.set_axis_off()
ax3.patch.set_alpha(0)
ax.text(8.3, -0.26, "anomaly map $h(\\\\cdot)$ (Renfrew survey)", ha="center",
        fontsize=7.6)"""),
    # graph: measurement annotation on the map factors
    ("""ax.text(XS[1] + 0.52, YZ + 0.04, "$\\\\bar{\\\\phi}^{\\\\mathrm{map}}_k$",
        fontsize=8.2, va="center", color=C_MAP)""",
     """ax.text(XS[1] + 0.52, YZ + 0.04, "$\\\\bar{\\\\phi}^{\\\\mathrm{map}}_k$",
        fontsize=8.2, va="center", color=C_MAP)
ax.annotate("scalar measurement $z_k$", xy=(XS[-1] + 0.13, YMAP + 0.02),
            xytext=(14.35, YMAP + 0.55), fontsize=7.0, color=C_MAP,
            arrowprops=dict(arrowstyle="-|>", color=C_MAP, lw=0.9))"""),
]

ERRHIST = [
    ("label=\"committed, 300 s lag (24.2 m)\")",
     "label=\"smoothed, 300 s lag (24.2 m)\")"),
]

LATLON = [
    ("label=\"proposed, committed\")", "label=\"proposed, smoothed\")"),
]


def main():
    patch("make_final_figures.py", MAIN)
    patch("make_errhist_figure.py", ERRHIST)
    patch("make_latlon_figure.py", LATLON)


if __name__ == "__main__":
    main()
