#!/usr/bin/env python3
"""Read the initial-position-uncertainty sweep, and locate the regime boundary.

  for PS in 0.1 100 300 1000 3000; do
      python3 mc_1007.py <line.h5> --seeds 30 --scale 0.1 --pos-sigma $PS \
              --norobust --out mc_1007_p$PS.csv
  done
  python3 pos_sweep_table.py <line.h5>

WHY THE SWEEP EXISTS. The export ships an initial horizontal position
uncertainty of 0.1 m. That is the TRACKING regime: one mode of the map
likelihood sits inside the prior, the posterior is unimodal, an EKF is optimal,
a particle filter has nothing to represent and relinearization has nothing to
fix. Every estimator comparison in this repository was run there, which is why
none of them separated. It is also not the operational case -- if position were
known to 0.1 m there would be no reason to switch the map on at all. A real
GNSS-denied cold start lets the INS drift first, so the uncertainty is
kilometre scale, and there the likelihood is genuinely multimodal.

The sweep applies the uncertainty to BOTH the prior the estimators receive and
the error actually drawn, with --scale held at 0.1 so the inertial states
contribute the same modest transient at every point and only the position knob
moves.

WHAT TO LOOK FOR. Not a monotone trend. The map on this segment has a 200 by
122 m grid step, so a displacement of 100 m does not leave the mode it started
in: it makes the problem harder without making it multimodal, and every
estimator degrades together. Separation should return once the displacement
crosses several correlation lengths. If the trough sits at the grid scale and
the recovery scales with it, the regime boundary is a property of the map
rather than of the estimator, which is the form a paper can state as a
condition rather than a tuning note.

The EKF+NN baseline cannot be added here: ekf_tlnn.jl is Julia and this runs in
a container without it. The committed eight-case table at 0.1 m has that column
and shows the network is not separable from the plain filter there anyway
(geometric mean 1.005, three wins of eight). Whether it holds up at kilometre
scale is untested and worth testing on a machine with Julia.
"""
import csv
import glob
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def sign_test(r):
    n = r.size
    w = int((r < 1).sum())
    k = min(w, n - w)
    return w, min(2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n, 1.0)


def main():
    files = sorted(glob.glob(os.path.join(HERE, "mc_1007_p*.csv")),
                   key=lambda p: float(re.search(r"_p([0-9.]+)\.csv", p).group(1)))
    if not files:
        print("no mc_1007_p*.csv -- run the sweep first (see docstring)")
        return 1
    grid = None
    if len(sys.argv) > 1:
        import h5py
        f = h5py.File(sys.argv[1], "r")
        gl, go = f["glat"][()], f["glon"][()]
        grid = (abs(gl[1] - gl[0]) * 6378137.0,
                abs(go[1] - go[0]) * 6378137.0 * math.cos(gl.mean()))
        print(f"map grid step {grid[0]:.0f} x {grid[1]:.0f} m, "
              f"anomaly rms {f['gh_rowmajor'][()].std():.0f} nT, "
              f"sigma {math.sqrt(f['R'][()]):.0f} nT")
    print()
    print("  ratio = ours / EKF on the same seed.  1.00 = tie, lower = we win.")
    print("  bar length is 1/ratio, so longer is better.")
    print()
    print(f"  {'init pos':>9}  {'INS':>7} {'EKF':>7}  |  "
          f"{'ours caus':>9} {'ratio':>6} {'wins':>6}   {'ours 300s':>9} "
          f"{'ratio':>6} {'wins':>6}")
    print("  " + "-" * 78)
    for p in files:
        ps = float(re.search(r"_p([0-9.]+)\.csv", p).group(1))
        r = list(csv.DictReader(open(p)))
        a = np.array([[float(x[k]) for k in
                       ("INS", "EKF", "FGO_causal", "FGO_smoothed")] for x in r])
        rc, rs = a[:, 2] / a[:, 1], a[:, 3] / a[:, 1]
        gc = math.exp(np.log(rc).mean())
        gs = math.exp(np.log(rs).mean())
        wc, pc = sign_test(rc)
        ws, _ = sign_test(rs)
        tie = "  <- tie" if pc > 0.05 else ""
        print(f"  {ps:8.0f}m  {np.median(a[:,0]):7.0f} {np.median(a[:,1]):7.0f}  |  "
              f"{np.median(a[:,2]):9.0f} {gc:6.3f} {wc:3d}/{rc.size:<3d}"
              f"  {np.median(a[:,3]):9.0f} {gs:6.3f} {ws:3d}/{rs.size:<3d}{tie}")
    print()
    print(f"  {'':9}  {'':15}   {'causal':<26}{'smoothed'}")
    for p in files:
        ps = float(re.search(r"_p([0-9.]+)\.csv", p).group(1))
        r = list(csv.DictReader(open(p)))
        a = np.array([[float(x[k]) for k in
                       ("EKF", "FGO_causal", "FGO_smoothed")] for x in r])
        gc = math.exp(np.log(a[:, 1] / a[:, 0]).mean())
        gs = math.exp(np.log(a[:, 2] / a[:, 0]).mean())
        print(f"  {ps:8.0f}m  {'':15}   {'#' * max(1, round(12 / gc / 1.2)):<26}"
              f"{'#' * max(1, round(12 / gs / 1.2))}")
    if "--plot" in sys.argv:
        plot(files, os.path.join(HERE, "pos_sweep.png"))
    if grid:
        print()
        print(f"the trough to watch for sits near the grid step, {min(grid):.0f} to "
              f"{max(grid):.0f} m: below it the draw stays inside one mode.")
    return 0


def plot(files, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs, ekf, cau, smo, rc_, rs_ = [], [], [], [], [], []
    for p in files:
        ps = float(re.search(r"_p([0-9.]+)\.csv", p).group(1))
        r = list(csv.DictReader(open(p)))
        a = np.array([[float(x[k]) for k in
                       ("EKF", "FGO_causal", "FGO_smoothed")] for x in r])
        xs.append(max(ps, 1.0))
        ekf.append(np.median(a[:, 0])); cau.append(np.median(a[:, 1]))
        smo.append(np.median(a[:, 2]))
        rc_.append(math.exp(np.log(a[:, 1] / a[:, 0]).mean()))
        rs_.append(math.exp(np.log(a[:, 2] / a[:, 0]).mean()))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(xs, ekf, "o-", label="EKF", color="#888")
    ax[0].loglog(xs, cau, "s-", label="ours, causal", color="#c0392b")
    ax[0].loglog(xs, smo, "^-", label="ours, 300 s lag", color="#2471a3")
    ax[0].set_xlabel("initial position uncertainty [m]")
    ax[0].set_ylabel("DRMS [m]")
    ax[0].set_title("absolute error")
    ax[0].grid(alpha=.3, which="both"); ax[0].legend()
    ax[1].semilogx(xs, rc_, "s-", label="causal / EKF", color="#c0392b")
    ax[1].semilogx(xs, rs_, "^-", label="300 s / EKF", color="#2471a3")
    ax[1].axhline(1.0, color="k", lw=.8)
    ax[1].axvspan(122, 200, color="orange", alpha=.15)
    ax[1].text(150, 1.02, "map grid", ha="center", fontsize=8, color="#a06000")
    ax[1].set_xlabel("initial position uncertainty [m]")
    ax[1].set_ylabel("ratio to EKF   (lower = better)")
    ax[1].set_title("where the estimator choice matters")
    ax[1].grid(alpha=.3); ax[1].legend()
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
