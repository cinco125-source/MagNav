#!/usr/bin/env python3
"""Sliding-window GTSAM FGO mirroring src/fgo_online.jl fgo_online_window:

  - windows of Lw = win/dt samples, stride = Lw - overlap/dt (Julia: 3000/2100)
  - each window is a batch LM (Huber robust) over the window's error states
  - each window COMMITS only its leading stride; the trailing overlap is
    smoother look-ahead re-processed by the next window
  - handoff = :filtered — the prior for window k+1 is the marginal at the
    commit boundary of a graph that has NOT seen the trailing overlap
    (solved on the truncated [i0, boundary] graph), so the overlap
    measurements are not double-counted

This is the structure the Julia reference actually runs (3 batch solves for
the 10-min line), unlike IncrementalFixedLagSmoother's per-epoch
marginalization (6000 incremental solves, hours in Python).
"""
import sys, math, time
import numpy as np
import h5py
import gtsam
from gtsam import symbol_shorthand
X = symbol_shorthand.X

R_EARTH = 6378137.0

def load(path):
    d = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            d[k] = f[k][()]
    return d

class Grid:
    def __init__(self, glat, glon, gh):
        self.glat = np.asarray(glat).ravel()
        self.glon = np.asarray(glon).ravel()
        # undo h5py's transpose of Julia's column-major grid (see run_gtsam.py)
        self.gh = np.asarray(gh).T
        assert self.gh.shape == (self.glat.size, self.glon.size)
        self.dlat = self.glat[1] - self.glat[0]
        self.dlon = self.glon[1] - self.glon[0]

    def _bilin(self, lat, lon):
        i = np.clip((lat - self.glat[0]) / self.dlat, 0, self.glat.size - 1.001)
        j = np.clip((lon - self.glon[0]) / self.dlon, 0, self.glon.size - 1.001)
        i0, j0 = int(i), int(j); fi, fj = i - i0, j - j0
        g = self.gh
        return (g[i0, j0]*(1-fi)*(1-fj) + g[i0+1, j0]*fi*(1-fj)
                + g[i0, j0+1]*(1-fi)*fj + g[i0+1, j0+1]*fi*fj)

    def value(self, lat, lon):
        return self._bilin(lat, lon)

    def grad(self, lat, lon):
        h = 1e-6
        dvlat = (self._bilin(lat+h, lon) - self._bilin(lat-h, lon)) / (2*h)
        dvlon = (self._bilin(lat, lon+h) - self._bilin(lat, lon-h)) / (2*h)
        return dvlat, dvlon


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "research/gtsam_poc/line_1007_06.h5"
    d = load(path)
    N   = int(d["N"]); nx = int(d["nx"]); nTL = int(d["nTL"])
    dt  = float(d["dt"]); Rm = float(d["R"]); warm = float(d["warm"])
    win = float(d["win"]); overlap = float(d["overlap"]); ref = float(d["ref_drms"])
    Phi = np.asarray(d["Phi"], dtype=float)
    if Phi.shape[0] == nx:
        Phi = np.moveaxis(Phi, 2, 0)
    else:
        # undo h5py's per-slice transpose of Julia's column-major tensor
        Phi = Phi.transpose(0, 2, 1)
    A = np.asarray(d["A"], dtype=float)
    if A.shape[0] != N: A = A.T
    meas = np.asarray(d["meas"], dtype=float).ravel()
    ins_lat = np.asarray(d["ins_lat"]).ravel(); ins_lon = np.asarray(d["ins_lon"]).ravel()
    true_lat = np.asarray(d["true_lat"]).ravel(); true_lon = np.asarray(d["true_lon"]).ravel()
    P0 = np.asarray(d["P0"], dtype=float); Qd_raw = np.asarray(d["Qd"], dtype=float)
    # scale-aware floor (see run_gtsam.py): position Qd diag is 1e-31 rad^2
    Qd = Qd_raw + 1e-20 * np.eye(nx)
    grid = Grid(d["glat"], d["glon"], d["gh"])

    iS  = nx - 1
    iTL = slice(17, 17 + nTL)

    dyn_nm   = gtsam.noiseModel.Gaussian.Covariance(Qd)
    meas_base = gtsam.noiseModel.Isotropic.Sigma(1, math.sqrt(Rm))
    meas_nm  = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345), meas_base)

    def dyn_err(Phi_t):
        def f(this, values, J):
            xa = values.atVector(this.keys()[0]); xb = values.atVector(this.keys()[1])
            e = xb - Phi_t @ xa
            if J is not None:
                J[0] = -Phi_t; J[1] = np.eye(nx)
            return e
        return f

    def meas_err(t):
        lat0, lon0, At, zt = ins_lat[t], ins_lon[t], A[t], meas[t]
        def f(this, values, J):
            x = values.atVector(this.keys()[0])
            lat, lon = lat0 + x[0], lon0 + x[1]
            h = grid.value(lat, lon) + At @ x[iTL] + x[iS]
            e = np.array([h - zt])
            if J is not None:
                jac = np.zeros((1, nx))
                glat_, glon_ = grid.grad(lat, lon)
                jac[0, 0] = glat_; jac[0, 1] = glon_
                jac[0, iTL] = At; jac[0, iS] = 1.0
                J[0] = jac
            return e
        return f

    def build_graph(i0, i1, x0_pr, P0_c):
        graph = gtsam.NonlinearFactorGraph()
        vals = gtsam.Values()
        graph.push_back(gtsam.PriorFactorVector(
            X(i0), x0_pr, gtsam.noiseModel.Gaussian.Covariance(P0_c)))
        for t in range(i0, i1 + 1):
            graph.add(gtsam.CustomFactor(meas_nm, [X(t)], meas_err(t)))
            vals.insert(X(t), x0_pr.copy())
            if t > i0:
                graph.add(gtsam.CustomFactor(dyn_nm, [X(t-1), X(t)], dyn_err(Phi[t-1])))
        return graph, vals

    def solve(graph, vals, max_iter=15):
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(max_iter)
        params.setLinearSolverType("MULTIFRONTAL_QR")   # conditioning, see run_gtsam.py
        return gtsam.LevenbergMarquardtOptimizer(graph, vals, params).optimize()

    Lw = max(2, round(win / dt))
    Lo = min(max(round(overlap / dt), 0), Lw - 1)
    stride = max(1, Lw - Lo)
    print(f"window Lw={Lw} overlap={Lo} stride={stride}", flush=True)

    est = np.zeros((N, nx))
    x0_pr = np.zeros(nx); P0_c = P0.copy()
    i0 = 0
    t_start = time.time()
    while i0 < N:
        i1 = min(i0 + Lw - 1, N - 1)
        graph, vals = build_graph(i0, i1, x0_pr, P0_c)
        sol = solve(graph, vals)
        gc1 = N - 1 if i1 == N - 1 else min(i0 + stride - 1, N - 1)
        for t in range(i0, gc1 + 1):
            est[t] = sol.atVector(X(t))
        print(f"window [{i0},{i1}] committed [{i0},{gc1}]  "
              f"({time.time()-t_start:.0f}s)", flush=True)
        if i1 == N - 1:
            break
        # :filtered handoff — marginal at the commit boundary of the graph
        # WITHOUT the trailing overlap, so the next window does not double-count
        tg, _ = build_graph(i0, gc1, x0_pr, P0_c)
        # warm-start the truncated solve from the window solution
        tv2 = gtsam.Values()
        for t in range(i0, gc1 + 1):
            tv2.insert(X(t), sol.atVector(X(t)))
        tsol = solve(tg, tv2, max_iter=10)
        # boundary covariance via a covariance-form forward (I)EKF pass, exactly
        # Julia's x_filt/P_filt handoff — this gtsam wheel's Marginals only
        # offers Cholesky, which the 1e30 conditioning breaks. The covariance
        # form tolerates the raw (singular-position) Qd like the Julia RTS.
        P = P0_c.copy()
        c_h = 1.345
        for t in range(i0, gc1 + 1):
            xb = tsol.atVector(X(t))
            lat, lon = ins_lat[t] + xb[0], ins_lon[t] + xb[1]
            glat_, glon_ = grid.grad(lat, lon)
            H = np.zeros((1, nx))
            H[0, 0] = glat_; H[0, 1] = glon_; H[0, iTL] = A[t]; H[0, iS] = 1.0
            resid = meas[t] - (grid.value(lat, lon) + A[t] @ xb[iTL] + xb[iS])
            e = abs(resid) / math.sqrt(Rm)
            wt = 1.0 if e <= c_h else max(c_h / e, 1e-6)   # Huber IRLS weight
            S = float(H @ P @ H.T) + Rm / wt
            K = (P @ H.T) / S
            P = P - K @ (H @ P)
            P = (P + P.T) / 2
            if t < gc1:
                P = Phi[t] @ P @ Phi[t].T + Qd_raw
        P0_c = (P + P.T) / 2
        x0_pr = tsol.atVector(X(gc1))
        i0 += stride

    est_lat = ins_lat + est[:, 0]; est_lon = ins_lon + est[:, 1]
    np.savez("research/gtsam_poc/est_window.npz", est=est, est_lat=est_lat,
             est_lon=est_lon, ins_lat=ins_lat, ins_lon=ins_lon,
             true_lat=true_lat, true_lon=true_lon)

    def drms_of(lat_, lon_, warm_v):
        m = (np.arange(N) * dt) >= warm_v
        if not m.any():
            return float("nan")
        dn = (lat_[m] - true_lat[m]) * R_EARTH
        de = (lon_[m] - true_lon[m]) * R_EARTH * np.cos(true_lat[m])
        return math.sqrt(np.mean(dn**2 + de**2))

    lines = []
    for wv in (0.0, 60.0, 120.0, 300.0, warm):
        g = drms_of(est_lat, est_lon, wv)
        i = drms_of(ins_lat, ins_lon, wv)
        lines.append(f"warm={wv:g}s  gtsam_window_drms={g:.2f} m  ins_drms={i:.2f} m")
        print(lines[-1], flush=True)
    print(f"(Julia FGO reference = {ref:.2f} m, exported warm={warm:g}s)", flush=True)
    with open("research/gtsam_poc/gtsam_poc_result_window.txt", "w") as f:
        f.write("\n".join(lines) + f"\njulia_ref {ref:.3f}\nN {N} nx {nx} nTL {nTL}\n")

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        with open("research/gtsam_poc/gtsam_poc_result_window.txt", "w") as f:
            f.write("PYTHON_ERROR\n" + tb)
        raise
