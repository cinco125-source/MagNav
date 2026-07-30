#!/usr/bin/env python3
"""Swap the breadth bar chart to a horizontal layout.

Eight grouped columns collide at column width; horizontal bars with the case
labels on the y axis read cleanly, matching the style of the earlier breadth
figure. Run once: python paper/replace_breadthbar.py
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "make_final_figures.py")

NEW = '''# ============================================================= fig_breadthbar
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

'''


def main():
    s = io.open(SRC, encoding="utf-8").read()
    mark = "# =============================================================== fig_lagsweep"
    a = s.find("# ============================================================= fig_breadthbar")
    b = s.index(mark)
    if a >= 0:
        s = s[:a] + NEW + s[b:]
    else:                       # block was lost earlier; insert fresh
        s = s[:b] + NEW + s[b:]
    io.open(SRC, "w", encoding="utf-8").write(s)
    print("breadthbar block in place")


if __name__ == "__main__":
    main()
