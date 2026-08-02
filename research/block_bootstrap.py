#!/usr/bin/env python3
"""How much DRMS can one flight line actually resolve?

A real SGL line is a single physical realization: the recorded INS cannot be
re-drawn, so a classical Monte-Carlo over it would be fabricated, which is why
the manuscript confines its MC to simulation. What a single realization does
support is a moving-block bootstrap: the position error is autocorrelated over
300 to 600 s, so resampling whole blocks preserves that structure and gives a
confidence interval on the DRMS.

The answer on line 1007.06 Mag 4, whole line, no warm-up, B=2000, 300 s blocks:

  EKF     72.6 m  [39.3, 86.6]
  EKF+NN  54.5 m  [33.7, 68.5]
  ours    60.96 m (point estimate; no committed error series for a CI)

The EKF's interval swallows our point estimate whole. A 16% gap on one line is
not resolvable when the line itself carries roughly plus or minus 30%, because
87 minutes at a 300 s correlation length is only about 17 independent blocks.

Two consequences worth carrying. Per-line DRMS printed to 0.1 m overstates what
one line knows by two orders of magnitude, so differences of a few metres
between estimators on a single line should not be argued. And aggregating
across lines, which is what the eight-case sign test does, is the right
instrument rather than a concession.

B matters less than it looks but not nothing: at B=30 the same data gives
[45.4, 81.2], visibly narrower than B=2000's [39.3, 86.6], because 30 replicates
under-resolve the tails. Resampling costs nothing -- no estimator is re-run --
so there is no reason to stop at 30.

PAIRED IS THE TEST THAT MATTERS. Comparing two wide intervals is the weakest
form of this. Both estimators fly the same trajectory over the same map, so
their errors share most of their structure; bootstrapping the per-block
DIFFERENCE cancels it and gives a far tighter interval on the margin itself.
That needs our error series, which lives in est_gtsam_ps100_<line>_m<mag>.npz
and is gitignored. Pass it with --est and the paired test runs.

Usage:
  block_bootstrap.py [tracks.csv] [--warm 0] [--blocks 300,600] [--B 2000]
                     [--est est_gtsam_ps100_1007_06_m4.npz]
"""
import os
import sys

import numpy as np

R_EARTH = 6378137.0
HERE = os.path.dirname(os.path.abspath(__file__))


def arg(flag, default, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def horiz(lat, lon, tlat, tlon):
    return np.hypot((lat - tlat) * R_EARTH, (lon - tlon) * R_EARTH * np.cos(tlat))


def boot(sq, n, L, B, rng):
    """Moving-block bootstrap of sqrt(mean(sq)) with overlapping blocks."""
    nb = max(1, n // L)
    out = np.empty(B)
    for b in range(B):
        s = rng.integers(0, max(1, n - L), size=nb)
        out[b] = np.sqrt(np.concatenate([sq[i:i + L] for i in s]).mean())
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else os.path.join(HERE, "baseline_tracks_1007_06_m4.csv")
    warm = float(arg("--warm", 0.0, float))
    blocks = [float(x) for x in arg("--blocks", "300,600").split(",")]
    B = int(arg("--B", 2000, int))
    est_path = arg("--est", None)
    rng = np.random.default_rng(33)

    d = np.genfromtxt(path, delimiter=",", names=True)
    t = d["t"] - d["t"][0]
    dt = float(np.median(np.diff(t)))
    m = t >= warm
    tlat, tlon = d["true_lat"][m], d["true_lon"][m]
    series = {}
    for lbl, p in (("INS", "ins"), ("EKF", "ekf"), ("EKF+NN", "nn")):
        if f"{p}_lat" in d.dtype.names:
            series[lbl] = horiz(d[p + "_lat"][m], d[p + "_lon"][m], tlat, tlon)

    if est_path:
        z = np.load(est_path)
        # decimated runs store the error state and the INS; sum them
        if "est_rt" in z and "ins_lat" in z:
            la = z["ins_lat"] + z["est_rt"][:, 0]; lo = z["ins_lon"] + z["est_rt"][:, 1]
            sl = z["ins_lat"] + z["est"][:, 0]; so = z["ins_lon"] + z["est"][:, 1]
            tl_, to_ = z["true_lat"], z["true_lon"]
        else:
            la, lo, sl, so = z["rt_lat"], z["rt_lon"], z["est_lat"], z["est_lon"]
            tl_, to_ = z["true_lat"], z["true_lon"]
        idx = z["idx"] if "idx" in z else np.arange(la.size)
        keep = (idx * dt) >= warm
        series["ours causal"] = horiz(la[keep], lo[keep], tl_[keep], to_[keep])
        series["ours 300 s"] = horiz(sl[keep], so[keep], tl_[keep], to_[keep])

    n_ref = min(len(v) for v in series.values())
    print(f"{os.path.basename(path)}  warm={warm:g}s  dt={dt:g}s  B={B}")
    print()
    for blk in blocks:
        L = int(blk / dt)
        print(f"--- block = {blk:g} s ({n_ref // L} blocks per replicate) ---")
        for lbl, e in series.items():
            sq = e ** 2
            L_ = int(blk / (dt if len(e) == len(t[m]) else
                            (t[m][-1] - t[m][0]) / max(1, len(e) - 1)))
            L_ = max(2, L_)
            v = boot(sq, len(e), L_, B, rng)
            lo_, hi_ = np.percentile(v, [2.5, 97.5])
            print(f"  {lbl:12s}{np.sqrt(sq.mean()):8.2f}  [{lo_:6.2f}, {hi_:6.2f}]"
                  f"   width {hi_-lo_:5.1f} m")
        # paired: bootstrap the per-block difference, which cancels the shared
        # trajectory and map error and is the only tight test available here
        if "ours causal" in series and "EKF" in series and len(series["EKF"]) == len(series["ours causal"]):
            a, b_ = series["EKF"] ** 2, series["ours causal"] ** 2
            nn_ = len(a); L2 = max(2, int(blk / dt)); nb = max(1, nn_ // L2)
            diff = np.empty(B)
            for k in range(B):
                s = rng.integers(0, max(1, nn_ - L2), size=nb)
                ia = np.concatenate([a[i:i + L2] for i in s])
                ib = np.concatenate([b_[i:i + L2] for i in s])
                diff[k] = np.sqrt(ib.mean()) / np.sqrt(ia.mean())
            lo_, hi_ = np.percentile(diff, [2.5, 97.5])
            print(f"  {'PAIRED ours/EKF':12s}{np.sqrt(b_.mean())/np.sqrt(a.mean()):8.2f}"
                  f"  [{lo_:6.2f}, {hi_:6.2f}]   <- the test that matters")
        print()
    if not est_path:
        print("no --est: our columns and the paired ratio are unavailable.")
        print("pass est_gtsam_ps100_<line>_m<mag>.npz to get them.")


if __name__ == "__main__":
    main()
