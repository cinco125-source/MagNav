#!/usr/bin/env python3
"""GTSAM cross-implementation / real-time figures for the new TAES sections
(cross-implementation validation, real-time incremental smoothing, and a
schematic of the MAP estimation spectrum of Sec. IV). Style follows
paper/fig_style.py (Paper-Orchestra plot_style guide) and the conventions
already used by paper/plot_tracks.py (INS = black dotted, proposed FGO =
blue solid, DRMS value folded into the legend label).

All numeric data are the measured GTSAM PoC results in
research/gtsam_poc/*.npz and *.txt (see research/gtsam_poc/README.md); this
script does not fabricate or interpolate any series. The one exception is
the Julia FGO segment reference (21.4 m), which is a *scalar* DRMS only --
no Julia time series was exported for this segment -- so it is drawn as a
horizontal reference line, not a trajectory; see the note written to
figs/fig_crossimpl_note.txt.

Run (needs numpy + matplotlib; no gtsam import required):
    wsl -e ~/gtsam_env/bin/python paper/make_gtsam_figures.py
  or, if a local Windows Python has matplotlib:
    python paper/make_gtsam_figures.py
-> paper/figs/fig_spectrum.pdf, fig_crossimpl.pdf, fig_realtime.pdf, fig_warmstart.pdf
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from fig_style import (apply_style, despine, COL_W,
                        C_PROPOSED, C_BASE1, C_BASE2, C_BASE3, C_REF, C_ACCENT)

apply_style()
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

# --- data locations (measured; nothing here is synthetic) ------------------
GTSAM_DIR = os.path.join(HERE, "..", "research", "gtsam_poc")
R_EARTH = 6378137.0
DT = 0.1  # s / sample

# full-line and segment reference numbers, all sourced from
# research/gtsam_poc/README.md and the *_result_*.txt CI outputs
JULIA_SEGMENT_DRMS = 21.4    # research/gtsam_poc/julia_ref.txt (warm=60 s)
JULIA_FULLLINE_DRMS = 13.8   # README headline table
GTSAM_FULLLINE_DRMS = 12.4   # README headline table


def _load(name):
    return np.load(os.path.join(GTSAM_DIR, name))


def herr(lat, lon, tlat, tlon):
    """Horizontal error [m] vs. the truth track (WGS-84 spherical approx.,
    matches research/gtsam_poc/plot_results.py and run_gtsam_window.py)."""
    dn = (lat - tlat) * R_EARTH
    de = (lon - tlon) * R_EARTH * np.cos(tlat)
    return np.hypot(dn, de)


def drms(lat, lon, tlat, tlon, t_s, warm_s=60.0):
    """DRMS after excluding the first warm_s seconds (matches drms_of() in
    research/gtsam_poc/run_gtsam_window.py / run_gtsam.py)."""
    m = t_s >= warm_s
    dn = (lat[m] - tlat[m]) * R_EARTH
    de = (lon[m] - tlon[m]) * R_EARTH * np.cos(tlat[m])
    return math.sqrt(np.mean(dn**2 + de**2))


# =============================================================================
# Figure 1 -- MAP estimation spectrum (schematic, no measured data)
# =============================================================================
def fig_spectrum():
    """Schematic of Sec. IV: the same factor graph solved at three points on
    the lag spectrum -- full-line batch (L=N), fixed-lag sliding window
    (L=Lw), and per-epoch incremental smoothing (L -> one epoch) -- sharing
    one time axis, with the lag L as the single design variable linking
    them (right-hand slider)."""
    INK, MUT, FAINT = "#1f2937", "#6b7280", "#aab2bd"
    BATCH_C, WIN_C, INC_C = C_ACCENT, C_PROPOSED, C_BASE2

    fig, ax = plt.subplots(figsize=(7.16, 3.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 112); ax.axis("off")

    x0, x1 = 9, 80           # shared timeline extent
    yB, yW, yI = 92, 55, 16  # row centers: batch, window, incremental
    N_EPOCH = 14
    xs = np.linspace(x0, x1, N_EPOCH)

    def row_label(y, letter, title, dy=12.5):
        ax.text(1.0, y + dy, f"({letter})", fontsize=8.5, fontweight="bold",
                 color=INK, ha="left", va="bottom")
        ax.text(6.2, y + dy, title, fontsize=8.0, color=INK,
                 ha="left", va="bottom")

    def epoch_ticks(y, color, alpha=0.55):
        for cx in xs:
            ax.plot([cx, cx], [y - 5.6, y - 4.4], color=color, lw=0.7,
                     alpha=alpha, zorder=2)

    # ---- (a) batch: one solve, spans the whole line, commits once at t=N --
    row_label(yB, "a", r"batch $(L=N)$ — offline bound")
    ax.add_patch(FancyBboxPatch((x0, yB - 6), x1 - x0, 12,
                 boxstyle="round,pad=0.3,rounding_size=1.6",
                 fc=BATCH_C, alpha=0.14, ec=BATCH_C, lw=1.4))
    ax.text((x0 + x1) / 2, yB, "solve once over all $N$ epochs",
             fontsize=7.6, color=BATCH_C, ha="center", va="center",
             fontweight="bold")
    epoch_ticks(yB, BATCH_C)
    ax.plot(x1, yB, "o", ms=6.5, color=BATCH_C, zorder=5)
    ax.annotate("commit at $t=N$\n(after the flight)", xy=(x1, yB - 6.2),
                xytext=(x1 - 1, yB - 15.5), fontsize=6.6, color=BATCH_C,
                ha="right", va="top",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=BATCH_C))

    # ---- (b) fixed-lag window: commit stride + overlap look-ahead + carry -
    row_label(yW, "b", r"fixed-lag window $(L=L_w)$ — quasi-online")
    Lw, stride = 24.0, 16.0
    starts = [x0, x0 + stride, x0 + 2 * stride]
    hh = 5.4
    for k, s in enumerate(starts):
        ax.add_patch(FancyBboxPatch((s, yW - hh), Lw, 2 * hh,
                     boxstyle="round,pad=0.10,rounding_size=1.1",
                     fc=WIN_C, alpha=0.16, ec=WIN_C, lw=1.2))
        ax.add_patch(Rectangle((s, yW - hh), stride, 2 * hh, fc=WIN_C,
                     alpha=0.30, ec="none"))
        ax.text(s + stride / 2, yW, "commit", fontsize=6.6, ha="center",
                 va="center", color="#0a3355")
        ax.text(s + stride + (Lw - stride) / 2, yW, "look-\nahead $L_o$",
                 fontsize=5.8, ha="center", va="center", color="#444")
        ax.text(s - 0.6, yW, r"$w_{%d}$" % (k + 1), fontsize=7.2, ha="right",
                 va="center", color=WIN_C)
    for a, b in ((0, 1), (1, 2)):
        ax.annotate("", xy=(starts[b], yW + hh + 2.6),
                     xytext=(starts[a] + stride, yW - hh - 2.6),
                     arrowprops=dict(arrowstyle="->", lw=1.0, color=C_BASE1))
    ax.text(starts[0] + stride + 0.3, yW + hh + 3.4,
             r"carry $(\hat{\boldsymbol{\chi}},\mathbf{P})$", fontsize=6.0,
             color=C_BASE1, ha="left")
    ax.text(starts[-1] + Lw + 1.5, yW, r"$\cdots$", fontsize=11,
             va="center", color=WIN_C)

    # ---- (c) per-epoch incremental: one solve per sample, marginalize -----
    row_label(yI, "c", r"incremental $(L\to\text{1 epoch})$ — real time")
    Lc = 6.0
    starts_c = np.linspace(x0, x1 - Lc, 11)
    for k, s in enumerate(starts_c):
        alpha = 0.10 if k < len(starts_c) - 1 else 0.30
        ax.add_patch(Rectangle((s, yI - 3.4), Lc, 6.8, fc=INC_C, alpha=alpha,
                     ec=INC_C if k == len(starts_c) - 1 else "none",
                     lw=1.1))
    ax.plot(starts_c[:-1] + 0.4, [yI - 4.6] * (len(starts_c) - 1), "x",
             ms=3.2, color=MUT, alpha=0.7, mew=0.8)
    ax.text((starts_c[0] + starts_c[3]) / 2, yI - 7.2, "marginalized states",
             fontsize=5.8, color=MUT, ha="center")
    ax.plot(starts_c[-1] + Lc, yI, "o", ms=5.5, color=INC_C, zorder=5)
    ax.annotate("causal estimate\nevery $\\Delta t$", xy=(starts_c[-1] + Lc, yI + 3.4),
                xytext=(x1 - 3, yI + 12), fontsize=6.6, color=INC_C,
                ha="right", va="bottom",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=INC_C))
    epoch_ticks(yI, INC_C, alpha=0.35)

    # ---- shared time axis --------------------------------------------------
    yT = 4
    ax.annotate("", xy=(x1 + 1.5, yT), xytext=(x0 - 1, yT),
                arrowprops=dict(arrowstyle="->", lw=1.0, color=INK))
    ax.text(x1 + 1.5, yT - 2.6, "time (epoch $t$)", fontsize=7.4, ha="right",
             color=INK)
    for cx, lbl in ((x0, "$t=1$"), (x1, "$t=N$")):
        ax.plot([cx, cx], [yT - 0.8, yT + 0.8], color=INK, lw=0.9)
        ax.text(cx, yT + 2.0, lbl, fontsize=6.6, ha="center", color=INK)

    # ---- right-hand lag "slider": the single design variable L -----------
    sx = 90
    ax.annotate("", xy=(sx, yI - 5), xytext=(sx, yB + 8),
                arrowprops=dict(arrowstyle="-", lw=5.5, color="0.88",
                                 shrinkA=0, shrinkB=0))
    for y, c in ((yB, BATCH_C), (yW, WIN_C), (yI, INC_C)):
        ax.plot(sx, y, "o", ms=6, color=c, zorder=6,
                 markeredgecolor="white", markeredgewidth=0.8)
    ax.annotate("", xy=(sx, yI - 9), xytext=(sx, yB + 12),
                arrowprops=dict(arrowstyle="->", lw=1.1, color=INK))
    ax.text(sx + 2.6, (yB + yI) / 2, "lag $L$\n(design variable)",
             fontsize=7.4, color=INK, ha="left", va="center", rotation=90)
    ax.text(sx, yB + 13.5, "$N$", fontsize=7.0, ha="center", color=BATCH_C)
    ax.text(sx, yW + 0.2, "$L_w$", fontsize=7.0, ha="center", va="bottom",
             color=WIN_C)
    ax.text(sx, yI - 10.5, "$1$", fontsize=7.0, ha="center", color=INC_C)
    ax.text(sx - 2.4, yB + 12.5, "more latency,\nmore accuracy", fontsize=5.7,
             color=MUT, ha="right", va="top", style="italic")
    ax.text(sx - 2.4, yI - 8.5, "less latency,\ncausal", fontsize=5.7,
             color=MUT, ha="right", va="bottom", style="italic")

    fig.savefig(os.path.join(OUT, "fig_spectrum.pdf"))
    fig.savefig(os.path.join(OUT, "fig_spectrum.png"), dpi=150)
    plt.close(fig)


# =============================================================================
# Figure 2 -- cross-implementation validation (measured GTSAM data)
# =============================================================================
def fig_crossimpl():
    """10-min segment (line 1007.06): GTSAM-window horizontal error time
    series vs. INS, against the two implementations' segment DRMS (GTSAM
    measured, Julia scalar-only)."""
    d = _load("est_window.npz")
    tlat, tlon = d["true_lat"], d["true_lon"]
    ilat, ilon = d["ins_lat"], d["ins_lon"]
    N = tlat.size
    t_min = np.arange(N) * DT / 60.0
    t_s = np.arange(N) * DT

    e_ins = herr(ilat, ilon, tlat, tlon)
    e_win = herr(d["est_lat"], d["est_lon"], tlat, tlon)
    ins_drms = drms(ilat, ilon, tlat, tlon, t_s)
    gtsam_drms = drms(d["est_lat"], d["est_lon"], tlat, tlon, t_s)

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.axvspan(0, 1.0, color="0.5", alpha=0.08, zorder=0)
    ax.axvline(1.0, color="0.5", lw=0.7, ls=":", zorder=1)

    # no shading of instantaneous error against the scalar Julia DRMS level:
    # that visual invites an apples-to-oranges reading (see fig_crossimpl_note.txt)
    ax.plot(t_min, e_ins, ":", color=C_REF, lw=1.3,
             label="INS (%.0f m DRMS)" % ins_drms)
    ax.plot(t_min, e_win, "-", color=C_PROPOSED, lw=1.7,
             label="GTSAM window (%.1f m)" % gtsam_drms)
    ax.axhline(JULIA_SEGMENT_DRMS, color=C_ACCENT, lw=1.1, ls="-.",
               label="Julia FGO window (%.1f m, DRMS only)" % JULIA_SEGMENT_DRMS)

    ax.annotate("60 s warm-up\n(excluded from DRMS)", xy=(1.0, 2),
                xytext=(1.3, 7), fontsize=6.2, color="0.4", ha="left",
                va="bottom")

    ax.set_xlabel("time [min]")
    ax.set_ylabel("horizontal error [m]")
    ax.set_xlim(t_min[0], t_min[-1])
    ax.set_ylim(0, max(e_ins.max(), JULIA_SEGMENT_DRMS) * 1.12)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92,
               edgecolor="0.7", fontsize=6.8, handlelength=1.8, borderpad=0.45)
    despine(ax)
    fig.savefig(os.path.join(OUT, "fig_crossimpl.pdf"))
    fig.savefig(os.path.join(OUT, "fig_crossimpl.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUT, "fig_crossimpl_note.txt"), "w") as f:
        f.write(
            "fig_crossimpl.pdf -- data provenance note (not for the tex; keep with the figure)\n"
            "=================================================================\n"
            "GTSAM curve: measured horizontal error time series, research/gtsam_poc/"
            "est_window.npz (line 1007.06, first 10-min segment, mag_5_uc cold start).\n"
            "GTSAM DRMS shown in the legend (%.2f m) is computed with the exact "
            "drms_of() formula in research/gtsam_poc/run_gtsam_window.py "
            "(warm=60 s exclusion) and matches research/gtsam_poc/"
            "gtsam_poc_result_window.txt line 2 (17.47 m).\n\n"
            "Julia curve: NOT plotted as a time series -- only a scalar segment DRMS "
            "(%.1f m) is available (research/gtsam_poc/julia_ref.txt, warm=60 s), "
            "because the Julia horizontal-error trajectory for this segment was not "
            "exported alongside the DRMS summary. It is shown as a horizontal "
            "reference line. Caption should say so explicitly, e.g.: \"the Julia "
            "curve is a scalar DRMS reference, not a time series, because only the "
            "summary statistic was exported for this run.\"\n\n"
            "Full-line reference (not shown on this segment plot, mention in text/"
            "caption if useful): GTSAM window %.1f m vs Julia %.1f m DRMS over the "
            "full 87.3-min line (README.md headline table).\n"
            % (gtsam_drms, JULIA_SEGMENT_DRMS, GTSAM_FULLLINE_DRMS, JULIA_FULLLINE_DRMS)
        )


# =============================================================================
# Figures 3 & 4 -- real-time incremental smoothing (measured GTSAM data)
# =============================================================================
def _load_realtime_series():
    d_batch = _load("est_batch.npz")
    d_l300 = _load("est_prog.npz")
    d_cold = _load("est_prog_lag30.npz")
    d_warm = _load("est_warm_lag30.npz")

    tlat, tlon = d_batch["true_lat"], d_batch["true_lon"]
    ilat, ilon = d_batch["ins_lat"], d_batch["ins_lon"]
    N = tlat.size
    t_min = np.arange(N) * DT / 60.0
    t_s = np.arange(N) * DT

    e_ins = herr(ilat, ilon, tlat, tlon)
    e_l300 = herr(d_l300["est_lat"], d_l300["est_lon"], tlat, tlon)
    lat_cold = ilat + d_cold["est_rt"][:, 0]; lon_cold = ilon + d_cold["est_rt"][:, 1]
    lat_warm = ilat + d_warm["est_rt"][:, 0]; lon_warm = ilon + d_warm["est_rt"][:, 1]
    e_cold = herr(lat_cold, lon_cold, tlat, tlon)
    e_warm = herr(lat_warm, lon_warm, tlat, tlon)

    d = dict(t_min=t_min, t_s=t_s, tlat=tlat, tlon=tlon,
             e_ins=e_ins, e_l300=e_l300, e_cold=e_cold, e_warm=e_warm,
             drms_ins=drms(ilat, ilon, tlat, tlon, t_s),
             drms_l300=drms(d_l300["est_lat"], d_l300["est_lon"], tlat, tlon, t_s),
             drms_cold=drms(lat_cold, lon_cold, tlat, tlon, t_s),
             drms_warm=drms(lat_warm, lon_warm, tlat, tlon, t_s),
             # converged-regime DRMS (t >= 300 s), the convention of paper
             # Sec. V "Real-time incremental smoothing" for post-lock-on accuracy
             drms300_ins=drms(ilat, ilon, tlat, tlon, t_s, warm_s=300.0),
             drms300_l300=drms(d_l300["est_lat"], d_l300["est_lon"], tlat, tlon,
                               t_s, warm_s=300.0),
             drms300_cold=drms(lat_cold, lon_cold, tlat, tlon, t_s, warm_s=300.0),
             drms300_warm=drms(lat_warm, lon_warm, tlat, tlon, t_s, warm_s=300.0))
    return d


def _lockon_time(t_min, e, thresh=30.0, t_max=6.5):
    """Last time within the first t_max minutes the error is still >=
    thresh -- a simple, reproducible proxy for the 'both runs lock in
    around the same [early] map feature' claim in
    research/gtsam_poc/README.md. Restricted to the early cold-start
    window so a later, unrelated low-gradient excursion (both curves dip
    back above 30 m briefly near t~8 min from ordinary map-gradient
    weakness, not the cold-start transient) is not mistaken for the
    lock-on event.
    """
    win = t_min <= t_max
    above = np.where(e[win] >= thresh)[0]
    return t_min[win][above[-1]] if len(above) else t_min[0]


def fig_realtime():
    """Real-time (causal) per-epoch ISAM2 estimates over the full 10-min
    segment: lag=30 s cold vs. warm TL prior vs. lag=300 s smoothed vs. INS,
    with the shared ~4.5-min lock-on marked. Legend is placed below the axes
    so it never overlaps the early cold-start transient."""
    d = _load_realtime_series()
    t = d["t_min"]

    lock_cold = _lockon_time(t, d["e_cold"])
    lock_warm = _lockon_time(t, d["e_warm"])
    lock = 0.5 * (lock_cold + lock_warm)

    fig, ax = plt.subplots(figsize=(COL_W, 3.05))
    fig.subplots_adjust(bottom=0.30)
    YMAX = 120.0
    # legend DRMS uses the converged regime (t >= 300 s), matching the paper's
    # Sec. V real-time text; the early transient is fig_warmstart's story
    ax.plot(t, d["e_ins"], ":", color=C_REF, lw=1.2,
             label="INS (%.0f m)" % d["drms300_ins"])
    ax.plot(t, d["e_cold"], "--", color=C_BASE3, lw=1.2,
             label="lag=30 s realtime, cold TL (%.1f m)" % d["drms300_cold"])
    ax.plot(t, d["e_warm"], "-.", color=C_BASE2, lw=1.2,
             label="lag=30 s realtime, warm TL (%.1f m)" % d["drms300_warm"])
    ax.plot(t, d["e_l300"], "-", color=C_PROPOSED, lw=1.7,
             label="lag=300 s, smoothed (%.1f m)" % d["drms300_l300"])
    ax.set_title("DRMS in legend: converged regime, $t\\geq$300 s",
                 fontsize=6.6, color="0.3", loc="right", pad=2)

    peak_xoff = {id(d["e_cold"]): (-1.0, 0.92), id(d["e_warm"]): (0.75, 0.74)}
    for e, c in ((d["e_cold"], C_BASE3), (d["e_warm"], C_BASE2)):
        pk = e.max()
        if pk > YMAX:
            ti = t[np.argmax(e)]
            dx, fy = peak_xoff[id(e)]
            ax.annotate("%.0f m" % pk, xy=(ti, YMAX), xytext=(ti + dx, YMAX * fy),
                         fontsize=6.2, color=c, ha="center", va="top",
                         arrowprops=dict(arrowstyle="-", lw=0.7, color=c))

    ax.axvline(lock, color="0.35", lw=0.8, ls=":", zorder=1)
    ax.annotate("lock-on $\\approx$%.1f min\n(map feature)" % lock,
                 xy=(lock, YMAX * 0.60), xytext=(lock + 0.35, YMAX * 0.70),
                 fontsize=6.4, color="0.3", ha="left")

    ax.set_xlabel("time [min]")
    ax.set_ylabel("horizontal error [m]")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, YMAX)
    ax.grid(True, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
               frameon=False, fontsize=6.6, handlelength=1.8,
               columnspacing=1.1, handletextpad=0.5)
    despine(ax)
    fig.savefig(os.path.join(OUT, "fig_realtime.pdf"))
    fig.savefig(os.path.join(OUT, "fig_realtime.png"), dpi=150)
    plt.close(fig)


def fig_warmstart():
    """Cold-start transient, zoomed to the first 6 minutes: a warm
    compensation prior roughly halves the DRMS-after-warm-up penalty
    (82.4 -> 49.5 m) but the map-gradient-observability transient itself is
    only removed once both runs lock on to the same map feature. Legend is
    placed below the axes; the DRMS callout sits top-left, clear of both."""
    d = _load_realtime_series()
    t_full = d["t_min"]
    m = t_full <= 6.0
    t = t_full[m]

    lock_cold = _lockon_time(t_full, d["e_cold"])
    lock_warm = _lockon_time(t_full, d["e_warm"])
    lock = 0.5 * (lock_cold + lock_warm)

    fig, ax = plt.subplots(figsize=(COL_W, 3.05))
    fig.subplots_adjust(bottom=0.30)
    YMAX = 110.0
    ax.axvspan(0, 1.0, color="0.5", alpha=0.08, zorder=0)
    ax.plot(t, d["e_ins"][m], ":", color=C_REF, lw=1.1, alpha=0.6,
             label="INS")
    ax.plot(t, d["e_l300"][m], "-", color=C_PROPOSED, lw=1.3, alpha=0.55,
             label="lag=300 s, smoothed")
    ax.plot(t, d["e_cold"][m], "--", color=C_BASE3, lw=1.6,
             label="lag=30 s realtime, cold TL")
    ax.plot(t, d["e_warm"][m], "-.", color=C_BASE2, lw=1.6,
             label="lag=30 s realtime, warm TL")

    peak_xoff = {id(d["e_cold"]): (-0.95, 0.90), id(d["e_warm"]): (0.55, 0.72)}
    for e_full, c in ((d["e_cold"], C_BASE3), (d["e_warm"], C_BASE2)):
        e = e_full[m]
        pk = e.max()
        if pk > YMAX:
            ti = t[np.argmax(e)]
            dx, fy = peak_xoff[id(e_full)]
            ax.annotate("%.0f m" % pk, xy=(ti, YMAX), xytext=(ti + dx, YMAX * fy),
                         fontsize=6.2, color=c, ha="center", va="top",
                         arrowprops=dict(arrowstyle="-", lw=0.7, color=c))

    ax.axvline(lock, color="0.35", lw=0.8, ls=":", zorder=1)
    ax.annotate("lock-on $\\approx$%.1f min" % lock, xy=(lock, YMAX * 0.20),
                 xytext=(lock + 0.15, YMAX * 0.28), fontsize=6.6, color="0.3",
                 ha="left")

    dt_txt = ("DRMS ($t\\geq$60 s, 0-10 min segment):\n"
              "cold %.1f m $\\rightarrow$ warm %.1f m  ($-$%.0f%%)"
              % (d["drms_cold"], d["drms_warm"],
                 100.0 * (1 - d["drms_warm"] / d["drms_cold"])))
    ax.text(0.97, 0.965, dt_txt, transform=ax.transAxes, fontsize=6.6,
             color="0.15", ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7",
                        lw=0.6, alpha=0.92))

    ax.set_xlabel("time [min]")
    ax.set_ylabel("horizontal error [m]")
    ax.set_xlim(0, 6.0)
    ax.set_ylim(0, YMAX)
    ax.grid(True, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
               frameon=False, fontsize=6.6, handlelength=1.8,
               columnspacing=1.1, handletextpad=0.5)
    despine(ax)
    fig.savefig(os.path.join(OUT, "fig_warmstart.pdf"))
    fig.savefig(os.path.join(OUT, "fig_warmstart.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_spectrum()
    fig_crossimpl()
    fig_realtime()
    fig_warmstart()
    print("wrote figures to", OUT)
    for f in sorted(os.listdir(OUT)):
        if f.startswith(("fig_spectrum", "fig_crossimpl", "fig_realtime", "fig_warmstart")):
            print("  ", f)
