#!/usr/bin/env python3
"""Local-minimum vs model-misspecification test for the Mag-4 divergence.

Builds the SAME decimated factor graph the fixed-lag smoother uses, but as one
full batch, and compares the objective value at several state trajectories:

  zeros        the cold-start linearization point (INS positions, no compensation)
  fixedlag     the diverged fixed-lag solution (loaded from its .npz)
  warm         constant Tolles-Lawson coefficients from a least-squares fit of
               (meas - map(INS)) onto A, zero navigation error
  LM(zeros)    batch Levenberg-Marquardt started from zeros
  LM(warm)     batch Levenberg-Marquardt started from warm

If LM(warm) reaches a LOWER objective than the diverged solution while also
having a small position error, the objective is right and the cold start is
landing in the wrong basin.  If the diverged solution has the lower objective,
the objective itself prefers the wrong trajectory (misspecification).

Usage: diag_basin.py <line.h5> [mag 4|5] [K=10] [--npz <est.npz>]
"""
import sys
import math
import time
import numpy as np
import h5py
import gtsam
from gtsam import symbol_shorthand

X = symbol_shorthand.X
R_EARTH = 6378137.0


class Grid:
    def __init__(self, glat, glon, gh):
        self.glat = np.asarray(glat).ravel()
        self.glon = np.asarray(glon).ravel()
        self.gh = np.asarray(gh)
        self.dlat = self.glat[1] - self.glat[0]
        self.dlon = self.glon[1] - self.glon[0]

    def _bilin(self, lat, lon):
        i = np.clip((lat - self.glat[0]) / self.dlat, 0, self.glat.size - 1.001)
        j = np.clip((lon - self.glon[0]) / self.dlon, 0, self.glon.size - 1.001)
        i0, j0 = int(i), int(j)
        fi, fj = i - i0, j - j0
        g = self.gh
        return (g[i0, j0] * (1 - fi) * (1 - fj) + g[i0 + 1, j0] * fi * (1 - fj)
                + g[i0, j0 + 1] * (1 - fi) * fj + g[i0 + 1, j0 + 1] * fi * fj)

    def value(self, lat, lon):
        return self._bilin(lat, lon)

    def grad(self, lat, lon):
        h = 1e-6
        return ((self._bilin(lat + h, lon) - self._bilin(lat - h, lon)) / (2 * h),
                (self._bilin(lat, lon + h) - self._bilin(lat, lon - h)) / (2 * h))


def main():
    path = sys.argv[1]
    mag = sys.argv[2] if len(sys.argv) > 2 else "4"
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    npz = (sys.argv[sys.argv.index("--npz") + 1]
           if "--npz" in sys.argv else None)

    d = {}
    with h5py.File(path, "r") as fh:
        for k in fh.keys():
            d[k] = fh[k][()]
    N = int(d["N"])
    nx = int(d["nx"])
    nTL = int(d["nTL"])
    dt = float(d["dt"])
    Rm = float(d["R"])
    warm_t = float(d["warm"])
    Phi = np.asarray(d["Phi"], dtype=float)
    A = np.asarray(d["A"], dtype=float)
    meas = np.asarray(d[f"meas_mag{mag}"], dtype=float).ravel()
    ins_lat = d["ins_lat"].ravel()
    ins_lon = d["ins_lon"].ravel()
    true_lat = d["true_lat"].ravel()
    true_lon = d["true_lon"].ravel()
    P0 = np.asarray(d["P0"], dtype=float)
    Qd = np.asarray(d["Qd"], dtype=float) + 1e-20 * np.eye(nx)
    grid = Grid(d["glat"], d["glon"], d["gh_rowmajor"])

    idx = np.arange(0, N, K)
    M = idx.size
    dtK = dt * K
    PhiK = np.zeros((M - 1, nx, nx))
    QdK = np.zeros((M - 1, nx, nx))
    for s in range(M - 1):
        P_ = np.eye(nx)
        Q_ = np.zeros((nx, nx))
        for t in range(idx[s], min(idx[s + 1], N - 1)):
            P_ = Phi[t] @ P_
            Q_ = Phi[t] @ Q_ @ Phi[t].T + Qd
        PhiK[s] = P_
        QdK[s] = (Q_ + Q_.T) / 2

    iS = nx - 1
    iTL = slice(17, 17 + nTL)

    meas_base = gtsam.noiseModel.Isotropic.Sigma(1, math.sqrt(Rm))
    meas_nm = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345), meas_base)
    prior_nm = gtsam.noiseModel.Gaussian.Covariance(P0)

    def dyn_err(Phi_t):
        def f(this, values, J):
            xa = values.atVector(this.keys()[0])
            xb = values.atVector(this.keys()[1])
            if J is not None:
                J[0] = -Phi_t
                J[1] = np.eye(nx)
            return xb - Phi_t @ xa
        return f

    def meas_err(t):
        lat0, lon0, At, zt = ins_lat[t], ins_lon[t], A[t], meas[t]

        def f(this, values, J):
            x = values.atVector(this.keys()[0])
            lat, lon = lat0 + x[0], lon0 + x[1]
            h = grid.value(lat, lon) + At @ x[iTL] + x[iS]
            if J is not None:
                jac = np.zeros((1, nx))
                gl, go = grid.grad(lat, lon)
                jac[0, 0] = gl
                jac[0, 1] = go
                jac[0, iTL] = At
                jac[0, iS] = 1.0
                J[0] = jac
            return np.array([h - zt])
        return f

    graph = gtsam.NonlinearFactorGraph()
    graph.push_back(gtsam.PriorFactorVector(X(0), np.zeros(nx), prior_nm))
    graph.add(gtsam.CustomFactor(meas_nm, [X(0)], meas_err(idx[0])))
    for s in range(1, M):
        graph.add(gtsam.CustomFactor(
            gtsam.noiseModel.Gaussian.Covariance(QdK[s - 1]),
            [X(s - 1), X(s)], dyn_err(PhiK[s - 1])))
        graph.add(gtsam.CustomFactor(meas_nm, [X(s)], meas_err(idx[s])))
    print(f"graph: {M} states, {graph.size()} factors", flush=True)

    il, io = ins_lat[idx], ins_lon[idx]
    tl, tn = true_lat[idx], true_lon[idx]
    ti = idx * dt

    def drms(est, warm_v=warm_t):
        m = ti >= warm_v
        dn = (il[m] + est[m, 0] - tl[m]) * R_EARTH
        de = (io[m] + est[m, 1] - tn[m]) * R_EARTH * np.cos(tl[m])
        return math.sqrt(np.mean(dn ** 2 + de ** 2))

    def to_values(est):
        v = gtsam.Values()
        for s in range(M):
            v.insert(X(s), est[s])
        return v

    def report(name, est, t_s=None):
        e = graph.error(to_values(est))
        extra = "" if t_s is None else f"  [{t_s:.0f}s]"
        print(f"{name:14s} error={e:16.4e}  drms={drms(est):10.1f} m{extra}",
              flush=True)
        return e

    # ---- candidate trajectories ----
    zeros = np.zeros((M, nx))
    report("zeros", zeros)

    # least-squares Tolles-Lawson fit of the uncompensated field at INS positions
    r0 = meas - np.array([grid.value(la, lo) for la, lo in zip(ins_lat, ins_lon)])
    beta, *_ = np.linalg.lstsq(A, r0, rcond=None)
    print(f"LS TL fit: |beta|_max={np.abs(beta).max():.1f}  bias={beta[-1]:.1f} nT",
          flush=True)
    warm = np.zeros((M, nx))
    warm[:, iTL] = beta
    report("warm", warm)

    if npz:
        z = np.load(npz)
        report("fixedlag", z["est"])

    params = gtsam.LevenbergMarquardtParams()
    params.setLinearSolverType("MULTIFRONTAL_QR")
    params.setMaxIterations(60)
    for name, init in (("LM(zeros)", zeros), ("LM(warm)", warm)):
        t0 = time.time()
        opt = gtsam.LevenbergMarquardtOptimizer(graph, to_values(init), params)
        res = opt.optimize()
        out = np.array([res.atVector(X(s)) for s in range(M)])
        report(name, out, time.time() - t0)
        np.savez(f"research/gtsam_poc/basin_{name.replace('(', '_').replace(')', '')}"
                 f"_m{mag}.npz", est=out, idx=idx, ins_lat=il, ins_lon=io,
                 true_lat=tl, true_lon=tn)


if __name__ == "__main__":
    main()
