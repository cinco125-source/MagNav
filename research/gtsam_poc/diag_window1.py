#!/usr/bin/env python3
"""Cold-start window sweep: batch-optimize only the FIRST W decimated states.

The fixed-lag smoother is already 400 m off after 30 s on Mag 4 while the INS is
2.6 m off, so the damage is done inside the first window. This script solves the
same graph as one small batch for a range of window lengths and reports the
objective and the position error, separating two explanations:

  - if the batch optimum of the first window is itself far from truth, the cold
    start is unobservable at that window length (a property of the problem);
  - if the batch optimum is close to truth, the fixed-lag machinery is at fault.

Usage: diag_window1.py <line.h5> [mag 4|5] [K=10] [W list, comma separated]
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
    Ws = ([int(w) for w in sys.argv[4].split(",")] if len(sys.argv) > 4
          else [30, 100, 300, 600, 1200])

    d = {}
    with h5py.File(path, "r") as fh:
        for k in fh.keys():
            d[k] = fh[k][()]
    N = int(d["N"])
    nx = int(d["nx"])
    nTL = int(d["nTL"])
    dt = float(d["dt"])
    Rm = float(d["R"])
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

    Wmax = max(Ws)
    idx = np.arange(0, min(N, (Wmax + 1) * K), K)
    M = idx.size
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
    # --tl-sigma S: widen the Tolles-Lawson block of the prior, which is the
    # control that decides whether the far-from-truth window solutions are a
    # property of the window or of a compensation prior scaled for a compensated
    # installation.
    sig_TL = 1.0
    if "--tl-sigma" in sys.argv:
        sig_TL = float(sys.argv[sys.argv.index("--tl-sigma") + 1])
        P0 = P0.copy()
        P0[iTL, iTL] = P0[iTL, iTL] * sig_TL ** 2
        print(f"TL prior sigma scaled by {sig_TL:g}", flush=True)
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

    params = gtsam.LevenbergMarquardtParams()
    params.setLinearSolverType("MULTIFRONTAL_QR")
    params.setMaxIterations(100)

    # --init-from <npz>: start the window solve at a known-good trajectory (the
    # full-line batch solution) instead of zeros. If that reaches a LOWER
    # objective than the cold start does, the short window has several minima
    # and the cold start is picking the wrong one; if it climbs to a HIGHER
    # objective, the short window's own optimum really is far from truth.
    init0 = None
    if "--init-from" in sys.argv:
        init0 = np.load(sys.argv[sys.argv.index("--init-from") + 1])["est"]
        print(f"initializing from {sys.argv[sys.argv.index('--init-from') + 1]}",
              flush=True)
    # --init-ridge: causal initialization. Fit the Tolles-Lawson coefficients to
    # (meas - map at the INS position) by ridge regression whose penalty is the
    # actual TL prior (sigma = 1 per coefficient, R on the residual), then start
    # every state from those coefficients with zero navigation error. The fit is
    # redone INSIDE each window length, over the raw samples of that window only,
    # so the initializer for a W-second solve never sees data past W seconds.
    ridge = "--init-ridge" in sys.argv

    def ridge_init(W):
        n_init = min(N, idx[W - 1] + 1)
        r0 = np.array([meas[t] - grid.value(ins_lat[t], ins_lon[t])
                       for t in range(n_init)])
        Aw = A[:n_init]
        beta = np.linalg.solve(Aw.T @ Aw + Rm * np.eye(nTL), Aw.T @ r0)
        out = np.zeros((W, nx))
        out[:, iTL] = beta
        print(f"  ridge TL init over {n_init*dt:.0f} s: "
              f"|beta|max={np.abs(beta).max():.1f}, residual rms "
              f"{np.sqrt(((r0 - Aw @ beta)**2).mean()):.1f} nT", flush=True)
        return out

    print(f"mag{mag} K={K}: batch optimum of the first W decimated states "
          f"(1 state = {dt*K:g} s)", flush=True)
    for W in Ws:
        graph = gtsam.NonlinearFactorGraph()
        graph.push_back(gtsam.PriorFactorVector(X(0), np.zeros(nx), prior_nm))
        graph.add(gtsam.CustomFactor(meas_nm, [X(0)], meas_err(idx[0])))
        for s in range(1, W):
            graph.add(gtsam.CustomFactor(
                gtsam.noiseModel.Gaussian.Covariance(QdK[s - 1]),
                [X(s - 1), X(s)], dyn_err(PhiK[s - 1])))
            graph.add(gtsam.CustomFactor(meas_nm, [X(s)], meas_err(idx[s])))
        start = ridge_init(W) if ridge else init0
        v = gtsam.Values()
        for s in range(W):
            v.insert(X(s), start[s] if start is not None else np.zeros(nx))
        e0 = graph.error(v)
        t0 = time.time()
        res = gtsam.LevenbergMarquardtOptimizer(graph, v, params).optimize()
        est = np.array([res.atVector(X(s)) for s in range(W)])
        e1 = graph.error(res)
        la = ins_lat[idx[:W]] + est[:, 0]
        lo = ins_lon[idx[:W]] + est[:, 1]
        dn = (la - true_lat[idx[:W]]) * R_EARTH
        de = (lo - true_lon[idx[:W]]) * R_EARTH * np.cos(true_lat[idx[:W]])
        drms = math.sqrt(np.mean(dn ** 2 + de ** 2))
        ins_dn = (ins_lat[idx[:W]] - true_lat[idx[:W]]) * R_EARTH
        ins_de = (ins_lon[idx[:W]] - true_lon[idx[:W]]) * R_EARTH * np.cos(true_lat[idx[:W]])
        ins_drms = math.sqrt(np.mean(ins_dn ** 2 + ins_de ** 2))
        tlb = est[-1, iTL]
        print(f"W={W:5d} ({W*dt*K:6.0f} s)  err {e0:11.4e} -> {e1:11.4e}   "
              f"drms {drms:9.1f} m (ins {ins_drms:6.1f})   "
              f"|TL|max {np.abs(tlb).max():8.1f}  S {est[-1, iS]:8.1f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
