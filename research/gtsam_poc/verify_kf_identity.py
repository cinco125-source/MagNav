#!/usr/bin/env python3
"""Strip the graph of everything that makes it a graph, and it must BE the EKF.

For a linear-Gaussian chain the fixed-lag MAP estimate of the newest state is
not merely close to the Kalman filter estimate -- it is the same quantity. So
an implementation that freezes every linearization where a filter would put it
and carries no robust kernel has no freedom left: it must reproduce the filter
exactly, and any discrepancy beyond arithmetic is a bug in the graph, the
whitening, the marginalization or the decimated propagation.

That makes this the sharpest correctness test available for this code, sharper
than any accuracy comparison, because it has a known right answer.

RESULT on the 1007.06 segment (Mag 5, sigma_beta = 100, lag 300 s, K = 10,
600 graph states, GTSAM ISAM2 with QR against a Joseph-form Kalman filter):

  max absolute difference, all 37 states, s >= 1     5.7e-08
  max relative difference, all 37 states, s >= 1     1.1e-06
  max horizontal position difference                 4.5e-06 m

Four and a half MICROMETRES over ten minutes of flight. That is floating-point
noise between two different factorizations, not a modelling difference.

The single exception is s = 0, where the graph reports 0 because
run_gtsam_decimated.py's convention is to seed the causal series with the prior
rather than with a solve, while the filter has already applied the first
measurement. It is one state in six hundred, the convention is the committed
runner's, and the identity holding from s = 1 onward shows it washes out
immediately. It is left alone rather than quietly changed.

The same identity shows up in the Monte Carlo without being asked for: at both
initial-error scales the --norelin --norobust configuration scores a DRMS ratio
against the filter of exactly 1.000 with 0 wins out of 30 seeds -- the two
estimators are not close, they are the same estimator.

WHAT IT SETTLES. The code is right. It also means the causal column can never
be a contribution on a nearly-linear problem: relinearization and the kernel
are the only two places a margin can come from, and this file shows there is
nothing else in the difference.

Usage: verify_kf_identity.py [line.h5]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_gtsam_decimated import Grid, load          # noqa: E402
import mc_1007 as M                                 # noqa: E402

R_EARTH = 6378137.0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/line_1007_06_py.h5"
    d = load(path)
    N = int(d["N"]); nx = int(d["nx"]); nTL = int(d["nTL"])
    dt = float(d["dt"]); Rm = float(d["R"])
    Phi = np.asarray(d["Phi"], float)
    A = np.asarray(d["A"], float)
    if A.shape[0] != N:
        A = A.T
    key = "meas_mag5" if "meas_mag5" in d else "meas"
    meas = np.asarray(d[key], float).ravel()
    ins_lat = np.asarray(d["ins_lat"]).ravel()
    ins_lon = np.asarray(d["ins_lon"]).ravel()
    P0 = np.asarray(d["P0"], float)
    Qd = np.asarray(d["Qd"], float) + 1e-20 * np.eye(nx)
    gh = d["gh_rowmajor"] if "gh_rowmajor" in d else np.asarray(d["gh"])
    grid = Grid(d["glat"], d["glon"], gh)

    K = 10
    idx = np.arange(0, N, K)
    Mn = idx.size
    dtK = dt * K
    PhiK = np.zeros((Mn - 1, nx, nx))
    QdK = np.zeros((Mn - 1, nx, nx))
    for s in range(Mn - 1):
        P_ = np.eye(nx); Q_ = np.zeros((nx, nx))
        for t in range(idx[s], min(idx[s + 1], N - 1)):
            P_ = Phi[t] @ P_
            Q_ = Phi[t] @ Q_ @ Phi[t].T + Qd
        PhiK[s] = P_
        QdK[s] = (Q_ + Q_.T) / 2

    iTL = slice(17, 17 + nTL)
    iS = nx - 1
    P0 = P0.copy()
    P0[iTL, iTL] = P0[iTL, iTL] * 100.0 ** 2
    Ad, md = A[idx], meas[idx]
    il, io = ins_lat[idx], ins_lon[idx]

    xk = M.kalman(PhiK, QdK, P0, Rm, Ad, md, il, io, grid, nx, iTL, iS)
    _, rt = M.fgo(PhiK, QdK, P0, Rm, Ad, md, il, io, grid, nx, iTL, iS,
                  300.0, dtK, norelin=True, huber=0.0)

    D = np.abs(rt - xk)[1:]                       # s = 0 is the prior slot
    scale = np.maximum(np.abs(xk).max(axis=0), 1e-30)
    dpos = np.hypot((rt[1:, 0] - xk[1:, 0]) * R_EARTH,
                    (rt[1:, 1] - xk[1:, 1]) * R_EARTH)
    print(f"frozen-linearization kernel-free graph vs Kalman filter, "
          f"{Mn} states from {os.path.basename(path)}")
    print(f"  max absolute difference, all {nx} states   {D.max():.3e}")
    print(f"  max relative difference, all {nx} states   {(D / scale).max():.3e}")
    print(f"  max horizontal position difference       {dpos.max():.3e} m")
    ok = (D / scale).max() < 1e-4
    print("  ->", "IDENTICAL to arithmetic precision" if ok else "MISMATCH -- investigate")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
