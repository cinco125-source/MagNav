#!/usr/bin/env python3
"""Data-driven and schematic figures for the manuscript (vector PDF).

Numbers come from the reproducible research/*.jl CI results; see paper/README.md.
Style follows paper/fig_style.py (Paper-Orchestra plot_style guide).
Run: python3 paper/make_figures.py  ->  paper/figs/*.pdf
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Patch, FancyArrowPatch
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox

from fig_style import (apply_style, despine, COL_W,
                       C_PROPOSED, C_BASE1, C_REF, C_ACCENT)

apply_style()
OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)


def fig_breadth():
    """Table IV as a grouped log-scale bar chart: two causal baselines
    (weak online-TL EKF, strong EKF+TL+NN) vs the NN-free FGO window. The
    marginalized-particle-filter baseline (MPF+TL) diverges past 10 km on every
    counted case, so it is reported in the table but not plotted here."""
    C_WEAK = "#9aa4b2"   # weak baseline (online-TL EKF), muted gray
    rows = [  # the four counted lines: FF (Flt1007) then SV (Flt1003).
        # The calibration line 1006.08 is set aside (see text) and not shown.
        # Numbers match Table IV / research/fgo_breadth_results.csv (committed CI run).
        # (label, EKF-online, EKF+TL+NN, FGO-win); None->div, "err"->off-map
        ("1007.06  M4  (FF)", 46.7, 48.8, 30.7), ("1007.06  M5  (FF)", 17.8, 18.3, 13.1),
        ("1007.02  M4  (FF)", None, 130.0, 39.9), ("1007.02  M5  (FF)", 31.6, 30.4, 15.4),
        ("1003.02  M4  (SV)", None, 101.0, 43.1), ("1003.02  M5  (SV)", 28.1, 29.5, 21.1),
        ("1003.08  M4  (SV)", "err", 46.6, 28.0), ("1003.08  M5  (SV)", 21.1, 20.7, 13.1),
    ]
    labels = [r[0] for r in rows]
    y = np.arange(len(rows))[::-1]
    h = 0.27
    fig, ax = plt.subplots(figsize=(COL_W, 2.95))
    DIVX = 3e4
    for i, (_, ek, nn, fg) in enumerate(rows):
        yy = y[i]
        if ek is None or ek == "err":
            lab = "diverged" if ek is None else "off-map err."
            ax.barh(yy + h, DIVX, height=h, color=C_WEAK, alpha=0.35,
                    hatch="////", edgecolor=C_WEAK, linewidth=0.5)
            ax.text(DIVX, yy + h, "  " + lab, va="center", ha="left",
                    fontsize=6.2, color="0.5", style="italic")
        else:
            ax.barh(yy + h, ek, height=h, color=C_WEAK, edgecolor="white",
                    linewidth=0.5)
        ax.barh(yy, nn, height=h, color=C_BASE1, edgecolor="white", linewidth=0.5)
        ax.barh(yy - h, fg, height=h, color=C_PROPOSED, edgecolor="white",
                linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlim(8, 1e5)
    ax.set_xlabel("horizontal DRMS [m]  (log scale)")
    ax.axvline(1e4, color="0.4", lw=0.7, ls=":")
    ax.text(1e4, len(rows)-0.3, "10 km", fontsize=6.5, color="0.4",
            ha="center", va="bottom")
    ax.grid(True, axis="x", which="major")
    # legend below the axis so it never sits on the (long) 1006.08 bars
    ax.legend(handles=[Patch(facecolor=C_WEAK, label="EKF, online TL (weak)"),
                       Patch(facecolor=C_BASE1, label="EKF+TL+NN (strong)"),
                       Patch(facecolor=C_PROPOSED, label="FGO window (proposed)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              frameon=False, fontsize=7, handlelength=1.3, columnspacing=1.2,
              handletextpad=0.5)
    despine(ax)
    fig.savefig(os.path.join(OUT, "fig_breadth.pdf"))
    plt.close(fig)


def fig_coldstart():
    """Line 1007.06 head-to-head vs the reimplemented EKF+TL+NN (our runs only)."""
    methods = ["FGO\nbatch", "FGO\nwin 5", "FGO win 5\n+Huber",
               "EKF+TL+NN\n(reimpl.)"]
    m4 = [123.7, 33.0, 30.7, 48.8]
    m5 = [68.1, 13.5, 13.1, 18.3]
    ours = [True, True, True, False]
    x = np.arange(len(methods)); w = 0.38
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    c = [C_PROPOSED if o else C_BASE1 for o in ours]
    b1 = ax.bar(x - w/2, m4, w, color=c, edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + w/2, m5, w, color=c, alpha=0.55, hatch="///",
                edgecolor="white", linewidth=0.6)
    for i, o in enumerate(ours):
        if not o:
            ax.axvspan(i - 0.5, i + 0.5, color="0.5", alpha=0.08, zorder=0)
    for b in (b1, b2):
        ax.bar_label(b, fmt="%.0f", fontsize=6.2, padding=1)
    ax.axvline(2.5, color="0.4", lw=0.7, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=7)
    ax.set_ylabel("horizontal DRMS [m]")
    ax.set_ylim(0, 142)
    ax.grid(True, axis="y")
    ax.legend(handles=[Patch(facecolor=C_PROPOSED, label="FGO (proposed)"),
                       Patch(facecolor=C_BASE1, label="EKF+TL+NN"),
                       Patch(facecolor="0.45", label="Mag 4"),
                       Patch(facecolor="0.45", alpha=0.55, hatch="///",
                             label="Mag 5")],
              loc="upper right", ncol=2, frameon=False, fontsize=6.8,
              columnspacing=1.0, handlelength=1.4)
    despine(ax)
    fig.savefig(os.path.join(OUT, "fig_coldstart.pdf"))
    plt.close(fig)


def fig_factorgraph():
    """Faithful factor graph of one window: INS error chain x_t (18 states),
    time-varying TL chain beta_t (random walk), joint prior, robust scalar-mag
    factors, factor parameters (INS mech., fluxgate, map) as dashed inputs,
    optional sensor-error variables theta. Full two-column width."""
    fig, ax = plt.subplots(figsize=(7.16, 3.15))
    ax.set_xlim(0, 15.6); ax.set_ylim(-0.15, 5.6); ax.axis("off")
    xs = [2.5, 5.3, 8.1, 10.9]          # epoch positions
    YX, YZ, YB = 4.15, 2.85, 1.55       # x-chain, z-factor, beta-chain rows
    R = 0.30

    def var(cx, cy, txt, fc, r=R, ec="k", ls="-"):
        ax.add_patch(Circle((cx, cy), r, fc=fc, ec=ec, lw=1.0, ls=ls, zorder=4))
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=8.5, zorder=5)

    def fac(cx, cy, fc="k", s=0.13):
        ax.add_patch(Rectangle((cx-s/2, cy-s/2), s, s, fc=fc, ec="k",
                     lw=0.6, zorder=4))

    def seg(x1, y1, x2, y2, **kw):
        ax.plot([x1, x2], [y1, y2], zorder=2,
                **{**dict(color="k", lw=0.9, ls="-"), **kw})

    # ---- variable chains -------------------------------------------------
    for i, cx in enumerate(xs):
        var(cx, YX, r"$\mathbf{x}_{%d}$" % (i+1), "#cfe0f3")
        var(cx, YB, r"$\boldsymbol{\beta}_{%d}$" % (i+1), "#f6d99b")
    ax.text(xs[-1]+1.15, YX, r"$\cdots$", fontsize=12, va="center")
    ax.text(xs[-1]+1.15, YB, r"$\cdots$", fontsize=12, va="center")
    seg(xs[-1]+R, YX, xs[-1]+0.9, YX); seg(xs[-1]+R, YB, xs[-1]+0.9, YB)

    # ---- process factors (Pinson-FOGM) and TL random walk ---------------
    for i in range(len(xs)-1):
        fx = (xs[i]+xs[i+1])/2
        fac(fx, YX); seg(xs[i]+R, YX, xs[i+1]-R, YX)
        fac(fx, YB); seg(xs[i]+R, YB, xs[i+1]-R, YB)
    ax.text((xs[0]+xs[1])/2, YX+0.42,
            r"$\|\mathbf{x}_{t+1}-\boldsymbol{\Phi}_t\mathbf{x}_t\|^2_{\mathbf{Q}_t^{-1}}$",
            fontsize=7.5, ha="center")
    ax.text((xs[0]+xs[1])/2, YB-0.48,
            r"$\|\boldsymbol{\beta}_{t+1}-\boldsymbol{\beta}_t\|^2_{(\mathbf{Q}^{\beta})^{-1}}$",
            fontsize=7.5, ha="center")

    # ---- joint prior on (x_1, beta_1) ------------------------------------
    fac(1.15, YZ, fc="0.35", s=0.16)
    seg(1.15, YZ+0.08, xs[0]-R*0.72, YX-R*0.72)
    seg(1.15, YZ-0.08, xs[0]-R*0.72, YB+R*0.72)
    ax.text(0.92, YZ-0.42, "joint prior\n$\\mathbf{P}_0$ / carried\nfrom window $k{-}1$",
            fontsize=6.6, ha="center", va="top", color="#333")

    # ---- measurement factors ---------------------------------------------
    for cx in xs:
        fac(cx, YZ, fc=C_PROPOSED)
        seg(cx, YX-R, cx, YZ+0.065); seg(cx, YZ-0.065, cx, YB+R)
    ax.annotate(
        r"$\rho(r_t/\sqrt{R}),\;\;r_t=z_t-h(\bar{\mathbf{p}}_t{+}\delta\mathbf{p}_t)-\mathbf{A}_t^{\top}\boldsymbol{\beta}_t-S_t$",
        xy=(xs[1]+0.09, YZ-0.06), xytext=(7.9, 0.42),
        fontsize=7.5, ha="center", va="center", color="#0a3355",
        arrowprops=dict(arrowstyle="->", lw=0.7, ls="--", color="#0a3355",
                        alpha=0.7))

    # ---- factor parameters (dashed inputs, not variables) ----------------
    ax.annotate(r"INS mech.: $\mathbf{f}^n,\,\mathbf{C}^n_b\rightarrow \boldsymbol{\Phi}_t,\mathbf{Q}_t$",
                xy=((xs[2]+xs[3])/2, YX+0.09), xytext=(12.15, 5.12),
                fontsize=6.8, color="0.25", ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.7, ls="--", color="0.45"))
    ax.annotate(r"fluxgate: $\mathbf{B}_t\rightarrow \mathbf{A}_t$;  map: $h,\ \mathbf{g}_t=\nabla h$",
                xy=(xs[3], YZ), xytext=(12.15, 2.85),
                fontsize=6.8, color="0.25", ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.7, ls="--", color="0.45"))

    # ---- state contents annotations --------------------------------------
    ax.text(0.1, 5.32, "INS error state  "
            r"$\mathbf{x}_t=[\,\delta\mathbf{p}\;\delta\mathbf{v}\;\boldsymbol{\psi}\;"
            r"h_a\;\hat{a}\;\mathbf{b}_a\;\mathbf{b}_g\;S\,]^{\top}\in\mathbb{R}^{18}$",
            fontsize=7.5, ha="left", color="#0a3355")
    ax.text(0.1, 0.02, "TL coefficients  "
            r"$\boldsymbol{\beta}_t\in\mathbb{R}^{n_\beta}$"
            "\n(permanent 3, induced 6, eddy 9, bias 1)",
            fontsize=7, ha="left", va="bottom", color="#7a5200")

    # ---- optional sensor-error variables ----------------------------------
    thx, thy = 13.6, 1.05
    var(thx, thy, r"$\boldsymbol{\theta}$", "#e9e9e9", r=0.27, ec="0.4", ls="--")
    for cx in xs[2:]:
        seg(thx-0.24, thy+0.14, cx+0.10, YZ-0.10, ls="--", color="0.55", lw=0.7)
    ax.text(thx+0.38, thy, "optional sensor-error\nvariables (not in this estimator):\n"
            r"$\{c_n\},\,b^{\mathrm{hi}},\,\gamma_0,\gamma_1$"
            "\n(couple to every $z_t$)",
            fontsize=6.4, ha="left", va="center", color="#333")

    fig.savefig(os.path.join(OUT, "fig_graph.pdf"))
    plt.close(fig)


def fig_window():
    """Sliding fixed-lag window: commit stride + overlap look-ahead + carry."""
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.1); ax.axis("off")
    ax.annotate("", xy=(9.9, 0.3), xytext=(0.1, 0.3),
                arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.text(9.85, 0.02, "time", fontsize=7.5, ha="right")
    Lw, stride = 4.0, 2.8
    ys = [2.35, 1.6, 0.9]
    starts = [0.1, 0.1+stride, 0.1+2*stride]
    for k, (s, yy) in enumerate(zip(starts, ys)):
        ax.add_patch(FancyBboxPatch((s, yy-0.24), Lw, 0.48,
                     boxstyle="round,pad=0.008", fc="#cfe0f3",
                     ec=C_PROPOSED, lw=1.1))
        ax.add_patch(Rectangle((s, yy-0.24), stride, 0.48, fc=C_PROPOSED,
                     alpha=0.32, ec="none"))
        ax.text(s+stride/2, yy, "commit", fontsize=7, ha="center",
                va="center", color="#0a3355")
        ax.text(s+stride+(Lw-stride)/2, yy, "look-\nahead", fontsize=6.2,
                ha="center", va="center", color="#444")
        ax.text(s-0.08, yy, r"$w_{%d}$" % (k+1), fontsize=8, ha="right",
                va="center")
    for a, b in ((0, 1), (1, 2)):
        ax.annotate("", xy=(starts[b], ys[b]+0.30),
                    xytext=(starts[a]+stride, ys[a]-0.30),
                    arrowprops=dict(arrowstyle="->", lw=0.9, color=C_BASE1))
    ax.text(starts[0]+stride+0.05, ys[0]+0.42, "carry state $+$ covariance",
            fontsize=6.6, color=C_BASE1, ha="left")
    fig.savefig(os.path.join(OUT, "fig_window.pdf"))
    plt.close(fig)


def fig_winlen():
    """DRMS vs window length (line 1007.06 cold start): the U-shaped tradeoff,
    swept densely from the reproducible research/fgo_winlen_results.csv."""
    p = os.path.join(RESEARCH, "fgo_winlen_results.csv")
    if os.path.exists(p):
        d = np.genfromtxt(p, delimiter=",", names=True,
                          dtype=None, encoding="utf-8")
        def series(tag):
            # windows shorter than the 5-min operating point under-determine the
            # calibration and are excluded (the 2-min window diverges on Mag 4).
            m = (d["mag"] == tag) & (d["win_min"] >= 5)
            w = d["win_min"][m]; y = d["drms"][m]
            o = np.argsort(w); return w[o], y[o]
        wl4, m4 = series("Mag 4"); wl5, m5 = series("Mag 5")
    else:  # fallback (pre-CI) — sparse points
        wl4 = wl5 = np.array([5.0, 87.0])
        m4 = np.array([30.7, 123.7]); m5 = np.array([13.1, 68.1])
    stat = float(max(wl4.max(), wl5.max()))   # static = longest window
    fig, ax = plt.subplots(figsize=(COL_W, 2.15))
    ax.plot(wl4, m4, "o-", color=C_PROPOSED, ms=4.8, lw=1.1, label="Mag 4", zorder=4)
    ax.plot(wl5, m5, "s-", color=C_BASE1, ms=4.4, lw=1.1, label="Mag 5", zorder=4)
    ax.set_xscale("log")
    ticks = sorted(set(np.r_[wl4, wl5].tolist()))
    ax.set_xticks(ticks)
    ax.set_xticklabels([("%d" % t) if t != stat else "%d\n(static)" % t
                        for t in ticks])
    ax.set_xlabel("window length $L_w$ [min]")
    ax.set_ylabel("horizontal DRMS [m]")
    ax.set_ylim(0, max(m4.max(), m5.max()) * 1.12)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", frameon=False)
    despine(ax)
    fig.savefig(os.path.join(OUT, "fig_winlen.pdf"))
    plt.close(fig)


def fig_obs():
    """Observability geometry: range(G) vs range(Psi) (Proposition 1, cond. iii).
    Soft filled span bands + origin marker; panel tags (a)/(b)."""
    fig, axs = plt.subplots(1, 2, figsize=(COL_W, 1.85))
    for a in axs:
        a.set_xlim(-1.3, 1.3); a.set_ylim(-1.15, 1.25); a.axis("off")
        a.set_aspect("equal")

    def span(a, ang, color, label, lx, ly, hw=0.055):
        th = np.deg2rad(ang)
        dx, dy = np.cos(th), np.sin(th)
        nx, ny = -dy*hw, dx*hw
        a.fill([-dx+nx, dx+nx, dx-nx, -dx-nx],
               [-dy+ny, dy+ny, dy-ny, -dy-ny],
               color=color, alpha=0.18, lw=0)
        a.plot([-dx, dx], [-dy, dy], color=color, lw=1.8,
               solid_capstyle="round")
        a.text(lx, ly, label, color=color, fontsize=7.5, ha="center")

    # (a) transversal: separable
    span(axs[0], 22, C_PROPOSED, r"range$(\mathbf{G})$", 0.88, 0.62)
    span(axs[0], 108, C_BASE1, r"range$(\boldsymbol{\Psi})$", -0.72, 0.92)
    axs[0].plot(0, 0, "o", color="k", ms=3.5, zorder=5)
    axs[0].annotate(r"$\{\mathbf{0}\}$", (0, 0), textcoords="offset points",
                    xytext=(7, -9), fontsize=7)
    axs[0].text(0, -1.08, "(a) separable", ha="center", fontsize=7.5,
                color="#333")

    # (b) near-collinear: confounded
    span(axs[1], 24, C_PROPOSED, r"range$(\mathbf{G})$", 1.02, -0.02)
    span(axs[1], 31, C_BASE1, r"range$(\boldsymbol{\Psi})$", -0.62, -0.80)
    d = np.deg2rad(27.5)
    axs[1].annotate("", xy=(1.05*np.cos(d), 1.05*np.sin(d)),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.1, color="0.25"))
    axs[1].text(-0.15, 0.62, "shared\ndirection", fontsize=6.4, color="0.25",
                ha="center")
    axs[1].plot(0, 0, "o", color="k", ms=3.5, zorder=5)
    axs[1].text(0, -1.08, "(b) confounded", ha="center", fontsize=7.5,
                color="#333")
    fig.savefig(os.path.join(OUT, "fig_obs.pdf"))
    plt.close(fig)


def fig_pipeline():
    """FGO concept (double column): sensors and the anomaly map feed a fixed-lag
    factor graph that carries navigation and online compensation as variables and
    is solved over a sliding window (detailed in Fig. 3), producing position and
    compensation. One accent (proposed blue), neutral grays; schematic."""
    INK, MUT, FAINT = "#1f2937", "#6b7280", "#aab2bd"
    FCN = "#eef2f7"; BLUE, ORANGE = C_PROPOSED, C_BASE1
    fig, ax = plt.subplots(figsize=(7.16, 2.35))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

    def box(x, y, w, h, txt, fc=FCN, ec=FAINT, fs=7.8, lw=1.0, tc=INK, ls="-"):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                     boxstyle="round,pad=0.2,rounding_size=1.2",
                     fc=fc, ec=ec, lw=lw, ls=ls))
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs, color=tc, zorder=6)

    def arrow(x0, y0, x1, y1, color=MUT, lw=1.2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=9, lw=lw, color=color, shrinkA=1, shrinkB=1))

    # inputs (left)
    box(11, 35, 19, 6.2, "INS / IMU\n$\\mathbf{f}^n,\\;\\mathbf{C}^n_b$")
    box(11, 26, 19, 6.2, "3-axis fluxgate\n$\\mathbf{B}_t\\!\\rightarrow\\!\\mathbf{A}_t$")
    box(11, 17, 19, 6.2, "scalar mag  $z_t$")
    box(11, 8, 19, 6.0, "anomaly map $h(\\cdot)$", fc="white", ls=(0, (3, 2)))

    # center: fixed-lag factor graph concept
    gx, gy, gw, gh = 31, 7, 33, 30
    ax.add_patch(FancyBboxPatch((gx, gy), gw, gh,
                 boxstyle="round,pad=0.3,rounding_size=1.6",
                 fc="white", ec=BLUE, lw=1.6))
    ax.text(gx+gw/2, gy+gh-2.6, "fixed-lag factor graph", ha="center", va="center",
            fontsize=8.2, color=BLUE, fontweight="bold")
    ax.text(gx+gw/2, gy+gh-5.6, "nav + online compensation as variables",
            ha="center", va="center", fontsize=6.0, color=MUT, style="italic")
    xs = np.linspace(gx+7, gx+gw-4.5, 5)
    ytop, ybot = gy+gh*0.44, gy+gh*0.23
    ax.plot(xs, [ytop]*5, "-", color=BLUE, lw=1.0, zorder=4)
    ax.plot(xs, [ybot]*5, "-", color=ORANGE, lw=1.0, zorder=4)
    for cx in xs:
        ax.plot([cx, cx], [ybot, ytop], "-", color=FAINT, lw=0.6, zorder=3)
        ax.plot(cx, ytop, "o", ms=4.2, color=BLUE, zorder=5)
        ax.plot(cx, ybot, "s", ms=3.8, color=ORANGE, zorder=5)
    ax.text(gx+5.6, ytop, "$\\mathbf{x}_t$", ha="right", va="center", fontsize=7, color=BLUE)
    ax.text(gx+5.6, ybot, "$\\boldsymbol{\\beta}_t$", ha="right", va="center", fontsize=7, color=ORANGE)
    ax.text(gx+gw/2, gy+2.2, "sliding-window solve \u2014 detail in Fig.\u20093",
            ha="center", va="center", fontsize=6.0, color=MUT)

    # right: window + outputs (side by side)
    box(84, 30, 22, 7.5, "fixed-lag window\ncommit, carry $(\\hat{\\boldsymbol{\\chi}},\\mathbf{P})$",
        fc="#e7eff9", ec=BLUE, tc="#0a3355", fs=7.0, lw=1.1)
    box(77.5, 15, 12.5, 6.4, "position\n$\\hat{\\mathbf{p}}_t$", ec=BLUE, tc=BLUE, fs=7.6)
    box(91.5, 15, 14.5, 6.4, "compensation\n$\\hat{\\boldsymbol{\\beta}}_t$", ec=ORANGE, tc=ORANGE, fs=7.6)

    # arrows in (horizontal into graph left edge)
    for yy in (35, 26, 17):
        arrow(20.5, yy, gx, yy)
    arrow(20.5, 8, gx, 10.5)
    arrow(gx+gw, gy+gh*0.5, 73, 30)
    arrow(80, 26.2, 77.5, 18.3)
    arrow(88, 26.2, 91.5, 18.3)

    fig.savefig(os.path.join(OUT, "fig_pipeline.pdf"))
    plt.close(fig)


def fig_concept():
    """Graphical abstract, three clean stages (schematic, no measured data):
    (1) one scalar reading couples position with the aircraft field; (2) a
    fixed-lag factor graph estimates both jointly, so later data corrects earlier
    states; (3) the payoff is bounded cold-start navigation, no neural network.
    Accent = proposed (blue); baseline = EKF (orange)."""
    INK, MUT, FAINT = "#1f2937", "#6b7280", "#aab2bd"
    FCN = "#eef2f7"; BLUE, ORANGE = C_PROPOSED, C_BASE1
    fig, ax = plt.subplots(figsize=(7.16, 2.15))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    def header(x, t):
        ax.text(x, 39.6, t, ha="center", va="top", fontsize=7.3, color=INK,
                fontweight="bold")

    def pill(x, y, w, h, txt, ec=FAINT, fc=FCN, tc=INK, fs=8.0, lw=1.0):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                     boxstyle="round,pad=0.15,rounding_size=1.2",
                     fc=fc, ec=ec, lw=lw))
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs, color=tc, zorder=6)

    def chev(x):
        ax.annotate("", xy=(x+2, 20), xytext=(x-2, 20),
                    arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=1.8))

    # Stage 1: coupling
    header(17, "one reading, two unknowns")
    pill(16, 32, 28, 5.2, "scalar magnetometer  $z_t$", ec=INK, fc="white", fs=8.4)
    parts = [("$z_t\\;=\\;$", INK), ("$h(\\mathbf{p}_t)$", BLUE),
             ("$\\,+\\,$", INK), ("$\\mathbf{A}_t^{\\top}\\boldsymbol{\\beta}_t$", ORANGE),
             ("$\\,+\\,\\eta_t$", MUT)]
    boxes = [TextArea(s, textprops=dict(color=c, fontsize=8.6)) for s, c in parts]
    ax.add_artist(AnnotationBbox(HPacker(children=boxes, align="baseline", pad=0, sep=1),
                  (16, 24.2), frameon=False, xycoords="data", box_alignment=(0.5, 0.5)))
    pill(7.8, 15, 13, 6.4, "position\n$\\mathbf{p}_t$", ec=BLUE, fc="white", tc=BLUE, fs=8)
    pill(24.5, 15, 15.5, 6.4, "aircraft field\n$\\mathbf{A}_t^{\\top}\\boldsymbol{\\beta}_t$",
         ec=ORANGE, fc="white", tc=ORANGE, fs=8)
    ax.annotate("", xy=(9.5, 18.5), xytext=(13.5, 21.4),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.3))
    ax.annotate("", xy=(23, 18.5), xytext=(18.5, 21.4),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.3))
    ax.text(16, 8.6, "coupled — no separate calibration", ha="center", va="center",
            fontsize=6.8, color=MUT, style="italic")

    chev(34)

    # Stage 2: joint estimation over a window
    header(55.5, "joint estimation on a fixed-lag graph")
    wx, wy, ww, wh = 39.5, 11, 33, 20
    ax.add_patch(FancyBboxPatch((wx, wy), ww, wh,
                 boxstyle="round,pad=0.25,rounding_size=1.4",
                 fc="white", ec=MUT, lw=1.2))
    ax.add_patch(Rectangle((wx+1.2, wy+1.2), ww*0.55, wh-2.4, fc=BLUE, alpha=0.08, ec="none"))
    ax.text(wx+ww*0.30, wy+wh-1.2, "commit", ha="center", va="top", fontsize=6.6, color="#0a3355")
    ax.text(wx+ww*0.80, wy+wh-1.2, "look-ahead", ha="center", va="top", fontsize=6.6, color=MUT)
    xs = np.linspace(wx+5, wx+ww-4, 6)
    ytop, ybot = wy+wh*0.60, wy+wh*0.30
    ax.plot(xs, [ytop]*6, "-", color=BLUE, lw=1.1, zorder=4)
    ax.plot(xs, [ybot]*6, "-", color=ORANGE, lw=1.1, zorder=4)
    for cx in xs:
        ax.plot([cx, cx], [ybot, ytop], "-", color=FAINT, lw=0.6, zorder=3)
        ax.plot(cx, ytop, "o", ms=5, color=BLUE, zorder=5)
        ax.plot(cx, ybot, "s", ms=4.6, color=ORANGE, zorder=5)
    ax.text(wx+3.0, ytop, "nav", ha="right", va="center", fontsize=6.6, color=BLUE)
    ax.text(wx+3.0, ybot, "comp.", ha="right", va="center", fontsize=6.6, color=ORANGE)
    ax.add_patch(FancyArrowPatch((xs[-1], (ytop+ybot)/2+1.4), (xs[1], (ytop+ybot)/2+1.4),
                 connectionstyle="arc3,rad=0.4", arrowstyle="-|>", mutation_scale=8,
                 lw=1.0, color=MUT))
    ax.text(wx+ww*0.5, wy+1.3, "later data corrects earlier states", ha="center",
            va="bottom", fontsize=6.6, color=MUT, style="italic")

    chev(75.5)

    # Stage 3: payoff
    header(89, "bounded cold start, no NN")
    px, py, pw, ph = 80.5, 11, 17.5, 18
    ax.add_patch(Rectangle((px, py), pw, ph, fc="white", ec=FAINT, lw=0.9))
    tt = np.linspace(0, 1, 100)
    ekf = py + 1.2 + (ph+4) * tt**2.3
    ax.plot(px+1.2+tt*(pw-2.4), np.clip(ekf, py, py+ph), "--", color=ORANGE, lw=1.5)
    ax.annotate("", xy=(px+1.2+(pw-2.4)*0.90, py+ph),
                xytext=(px+1.2+(pw-2.4)*0.84, py+ph-2.6),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))
    fgo = py + 2.8 - 0.7*np.cos(tt*3.2)*np.exp(-tt*1.8)
    ax.plot(px+1.2+tt*(pw-2.4), fgo, "-", color=BLUE, lw=2.0)
    ax.text(px+pw*0.50, py+ph-1.2, "online-TL EKF", ha="center", va="top",
            fontsize=6.3, color=ORANGE)
    ax.text(px+pw-1.2, py+3.4, "proposed", ha="right", va="bottom",
            fontsize=6.5, color=BLUE)
    ax.annotate("", xy=(px+pw, py), xytext=(px, py),
                arrowprops=dict(arrowstyle="-|>", color=MUT, lw=0.9))
    ax.annotate("", xy=(px, py+ph), xytext=(px, py),
                arrowprops=dict(arrowstyle="-|>", color=MUT, lw=0.9))
    ax.text(px+pw-0.5, py-0.8, "time", ha="right", va="top", fontsize=6.2, color=MUT)
    ax.text(px-0.6, py+ph-0.3, "error", ha="right", va="top", fontsize=6.2, color=MUT, rotation=90)

    fig.savefig(os.path.join(OUT, "fig_concept.pdf"))
    plt.close(fig)


RESEARCH = os.path.join(os.path.dirname(__file__), "..", "research")


def _running_mean(y, w):
    w = max(1, int(w)); k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


def fig_consistency():
    """Monte-Carlo NEES consistency: 2-DOF horizontal-position NEES vs time for
    the causal EKF and the batch FGO, against the chi-square 95% band. Reads the
    simulated-flight Monte-Carlo output (research/montecarlo_nees.csv)."""
    p = os.path.join(RESEARCH, "montecarlo_nees.csv")
    if not os.path.exists(p):
        print("  [skip fig_consistency: montecarlo_nees.csv not found]"); return
    d = np.genfromtxt(p, delimiter=",", names=True)
    t = d["t"] / 60.0
    lo, hi, ideal = 0.0506, 7.378, 2.0   # chi2_2 95% two-sided, E[NEES]=2
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    ax.axhspan(lo, hi, color="0.85", zorder=0, label="95% band ($\\chi^2_2$)")
    ax.axhline(ideal, color="0.45", lw=0.8, ls=":", zorder=1)
    w = max(5, len(t) // 60)
    ax.plot(t, _running_mean(d["nees_ekf"], w), "-", color=C_BASE1, lw=1.3,
            label="EKF (causal)")
    if "nees_mpf" in d.dtype.names:
        ax.plot(t, _running_mean(d["nees_mpf"], w), "-", color=C_ACCENT, lw=1.3,
                label="MPF (particle)")
    ax.plot(t, _running_mean(d["nees_fgo"], w), "-", color=C_PROPOSED, lw=1.7,
            label="FGO (batch)")
    ax.set_yscale("log")
    ax.set_xlabel("time [min]")
    ax.set_ylabel("position NEES (2 DOF)")
    ax.set_xlim(t[0], t[-1])
    ax.text(t[-1], ideal, " ideal 2", fontsize=6.5, color="0.4", va="center")
    ax.legend(loc="upper right", frameon=False, fontsize=6.6, ncol=2)
    despine(ax)
    fig.savefig(os.path.join(OUT, "fig_consistency.pdf"))
    plt.close(fig)


def fig_sigma():
    """+/-2 sigma covariance envelope on a representative run: North and East
    position error vs time inside the estimator +/-2 sigma bands (FGO). Reads
    research/montecarlo_sigma.csv."""
    p = os.path.join(RESEARCH, "montecarlo_sigma.csv")
    if not os.path.exists(p):
        print("  [skip fig_sigma: montecarlo_sigma.csv not found]"); return
    d = np.genfromtxt(p, delimiter=",", names=True)
    t = d["t"] / 60.0
    fig, axs = plt.subplots(2, 1, figsize=(COL_W, 2.9), sharex=True)
    for ax, err, std, lab in ((axs[0], d["n_err"], d["n_std"], "North"),
                              (axs[1], d["e_err"], d["e_std"], "East")):
        ax.fill_between(t, -2*std, 2*std, color=C_PROPOSED, alpha=0.15,
                        label="$\\pm2\\sigma$")
        ax.plot(t, err, "-", color=C_PROPOSED, lw=1.1, label="error")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_ylabel(f"{lab} err [m]")
        despine(ax)
    axs[0].legend(loc="upper right", frameon=False, fontsize=6.6, ncol=2)
    axs[1].set_xlabel("time [min]")
    axs[1].set_xlim(t[0], t[-1])
    fig.savefig(os.path.join(OUT, "fig_sigma.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_breadth(); fig_coldstart(); fig_factorgraph(); fig_window(); fig_pipeline()
    fig_winlen(); fig_obs(); fig_concept(); fig_consistency(); fig_sigma()
    print("wrote figures to", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)
