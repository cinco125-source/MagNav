#!/usr/bin/env python3
"""Monte-Carlo consistency (NEES) of the incremental fixed-lag smoother.

Perfect-model consistency test on the real line geometry: the error-state
truth is simulated from the same Phi/Qd the estimator uses, measurements are
synthesized from the same map, regressor and R, and the estimator runs exactly
as in run_gtsam_decimated.py (sigma_beta = 100, lag 300 s, 1 Hz states, Huber
off for the test since the noise is truly Gaussian). For each retained state
about to leave the lag, the smoothed position NEES
    e' P^{-1} e,  e = estimated - true position error  (2 DOF)
is recorded, with P the iSAM2 marginal covariance; the causal NEES at the
newest state is recorded likewise. ANEES with chi-square bounds comes from
aggregating over runs.

Usage: mc_nees.py <line.h5> [runs=20] [seed0=1000] [tag]
Writes research/gtsam_poc/mc_nees_result<tag>.csv (one row per run) and prints
the aggregate.
"""
import math
import sys
import time

import gtsam
import gtsam_unstable
import h5py
import numpy as np
from gtsam import symbol_shorthand

X = symbol_shorthand.X
R_EARTH = 6378137.0
SIG_TL = 100.0
LAG = 300.0
K = 10


class Grid:
    def __init__(self, glat, glon, gh):
        self.glat = np.asarray(glat).ravel()
        self.glon = np.asarray(glon).ravel()
        self.gh = np.asarray(gh)
        self.dlat = self.glat[1] - self.glat[0]
        self.dlon = self.glon[1] - self.glon[0]

    def value(self, lat, lon):
        i = np.clip((lat - self.glat[0]) / self.dlat, 0, self.glat.size - 1.001)
        j = np.clip((lon - self.glon[0]) / self.dlon, 0, self.glon.size - 1.001)
        i0, j0 = int(i), int(j)
        fi, fj = i - i0, j - j0
        g = self.gh
        return (g[i0, j0] * (1 - fi) * (1 - fj) + g[i0 + 1, j0] * fi * (1 - fj)
                + g[i0, j0 + 1] * (1 - fi) * fj + g[i0 + 1, j0 + 1] * fi * fj)

    def grad(self, lat, lon):
        h = 1e-6
        return ((self.value(lat + h, lon) - self.value(lat - h, lon)) / (2 * h),
                (self.value(lat, lon + h) - self.value(lat, lon - h)) / (2 * h))


def main():
    path = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    tag = sys.argv[4] if len(sys.argv) > 4 else ""

    d = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            d[k] = f[k][()]
    N = int(d["N"]); nx = int(d["nx"]); nTL = int(d["nTL"])
    dt = float(d["dt"]); Rm = float(d["R"])
    Phi = np.asarray(d["Phi"], dtype=float)
    A = np.asarray(d["A"], dtype=float)
    ins_lat = d["ins_lat"].ravel(); ins_lon = d["ins_lon"].ravel()
    P0 = np.asarray(d["P0"], dtype=float).copy()
    Qd = np.asarray(d["Qd"], dtype=float) + 1e-20 * np.eye(nx)
    grid = Grid(d["glat"], d["glon"], d["gh_rowmajor"])

    iTL = slice(17, 17 + nTL)
    iS = nx - 1
    P0[iTL, iTL] = P0[iTL, iTL] * SIG_TL ** 2
    L0 = np.linalg.cholesky(P0)
    LQ = np.linalg.cholesky(Qd)

    idx = np.arange(0, N, K)
    M = idx.size
    dtK = dt * K
    ell = int(round(LAG / dtK))

    PhiK = np.zeros((M - 1, nx, nx))
    QdK = np.zeros((M - 1, nx, nx))
    for s in range(M - 1):
        P_ = np.eye(nx); Q_ = np.zeros((nx, nx))
        for t in range(idx[s], min(idx[s + 1], N - 1)):
            P_ = Phi[t] @ P_
            Q_ = Phi[t] @ Q_ @ Phi[t].T + Qd
        PhiK[s] = P_; QdK[s] = (Q_ + Q_.T) / 2
    LQK = [np.linalg.cholesky(QdK[s] + 1e-24 * np.eye(nx)) for s in range(M - 1)]

    dyn_nms = [gtsam.noiseModel.Gaussian.Covariance(QdK[s]) for s in range(M - 1)]
    prior_nm = gtsam.noiseModel.Gaussian.Covariance(P0)
    meas_nm = gtsam.noiseModel.Isotropic.Sigma(1, math.sqrt(Rm))

    def dyn_err(Phi_t):
        def f(this, values, J):
            xa = values.atVector(this.keys()[0])
            xb = values.atVector(this.keys()[1])
            if J is not None:
                J[0] = -Phi_t; J[1] = np.eye(nx)
            return xb - Phi_t @ xa
        return f

    rows = []
    for run in range(runs):
        rng = np.random.default_rng(seed0 + run)
        # ---- simulate the error-state truth at the kept cadence ----------
        chi = np.zeros((M, nx))
        chi[0] = L0 @ rng.standard_normal(nx)
        for s in range(M - 1):
            chi[s + 1] = PhiK[s] @ chi[s] + LQK[s] @ rng.standard_normal(nx)
        # ---- synthesize measurements at kept samples ---------------------
        z = np.empty(M)
        for s in range(M):
            t = idx[s]
            lat = ins_lat[t] + chi[s, 0]; lon = ins_lon[t] + chi[s, 1]
            z[s] = (grid.value(lat, lon) + A[t] @ chi[s, iTL] + chi[s, iS]
                    + math.sqrt(Rm) * rng.standard_normal())

        def meas_err(s):
            t = idx[s]
            lat0, lon0, At, zt = ins_lat[t], ins_lon[t], A[t], z[s]

            def f(this, values, J):
                x = values.atVector(this.keys()[0])
                lat, lon = lat0 + x[0], lon0 + x[1]
                h = grid.value(lat, lon) + At @ x[iTL] + x[iS]
                if J is not None:
                    jac = np.zeros((1, nx))
                    gl, go = grid.grad(lat, lon)
                    jac[0, 0] = gl; jac[0, 1] = go
                    jac[0, iTL] = At; jac[0, iS] = 1.0
                    J[0] = jac
                return np.array([h - zt])
            return f

        # ---- run the incremental fixed-lag smoother ----------------------
        params = gtsam.ISAM2Params()
        params.setFactorization("QR")
        sm = gtsam_unstable.IncrementalFixedLagSmoother(LAG, params)
        KTM = gtsam_unstable.FixedLagSmootherKeyTimestampMap

        graph = gtsam.NonlinearFactorGraph(); vals = gtsam.Values(); ts = KTM()
        graph.push_back(gtsam.PriorFactorVector(X(0), np.zeros(nx), prior_nm))
        graph.add(gtsam.CustomFactor(meas_nm, [X(0)], meas_err(0)))
        vals.insert(X(0), np.zeros(nx)); ts.insert((X(0), 0.0))
        est_rt = np.zeros((M, nx))

        nees_s, nees_c = [], []
        t0 = time.time()
        for s in range(1, M):
            graph.add(gtsam.CustomFactor(dyn_nms[s - 1], [X(s - 1), X(s)],
                                         dyn_err(PhiK[s - 1])))
            graph.add(gtsam.CustomFactor(meas_nm, [X(s)], meas_err(s)))
            x_init = PhiK[s - 1] @ est_rt[s - 1] if s > 1 else np.zeros(nx)
            vals.insert(X(s), x_init); ts.insert((X(s), s * dtK))
            sm.update(graph, vals, ts)
            graph = gtsam.NonlinearFactorGraph()
            vals = gtsam.Values(); ts = KTM()
            cur = sm.calculateEstimate()
            est_rt[s] = cur.atVector(X(s))
            isam = sm.getISAM2()
            # causal NEES at the newest state (position block)
            e_c = est_rt[s][:2] - chi[s, :2]
            Pc = isam.marginalCovariance(X(s))[:2, :2]
            nees_c.append(float(e_c @ np.linalg.solve(Pc, e_c)))
            # smoothed NEES for the state about to leave the lag
            k_old = s - ell
            if k_old >= 1 and cur.exists(X(k_old)):
                e_s = cur.atVector(X(k_old))[:2] - chi[k_old, :2]
                Ps = isam.marginalCovariance(X(k_old))[:2, :2]
                nees_s.append(float(e_s @ np.linalg.solve(Ps, e_s)))
        el = time.time() - t0
        rows.append((run, seed0 + run, np.mean(nees_c), np.mean(nees_s),
                     len(nees_s), el))
        print(f"run {run:2d}: causal ANEES {np.mean(nees_c):6.3f}  "
              f"smoothed ANEES {np.mean(nees_s):6.3f}  "
              f"(n={len(nees_s)}, {el:.0f}s)", flush=True)

    import csv
    out = f"research/gtsam_poc/mc_nees_result{tag}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "seed", "anees_causal", "anees_smoothed",
                    "n_smoothed", "t_s"])
        w.writerows(rows)
    ac = np.array([r[2] for r in rows]); asm = np.array([r[3] for r in rows])
    print(f"\naggregate over {runs} runs (2-DOF position, expect 2.0):")
    print(f"  causal   ANEES {ac.mean():.3f}  (run spread {ac.min():.2f}"
          f"-{ac.max():.2f})")
    print(f"  smoothed ANEES {asm.mean():.3f}  (run spread {asm.min():.2f}"
          f"-{asm.max():.2f})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
