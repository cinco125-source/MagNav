#!/usr/bin/env python3
"""Warm-start variant of the per-epoch incremental smoother (run_gtsam.py):
seed the TL prior from a converged solution (stand-in for a previous
calibration line) instead of the cold zeros/sigma=1 prior, then run the
lag=30 s ISAM2 fixed-lag smoother. Shows how much of the cold-start
transient is the in-flight TL calibration.

Usage: run_gtsam_warm.py <line.h5> <est_npz_with_converged_TL> [lag_s=30]
"""
import sys, math, time
import numpy as np
import h5py
import gtsam
import gtsam_unstable
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
        self.gh = np.asarray(gh).T          # undo h5py transpose (see run_gtsam.py)
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
    path = sys.argv[1]
    npz_path = sys.argv[2]
    lag = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0

    d = load(path)
    N   = int(d["N"]); nx = int(d["nx"]); nTL = int(d["nTL"])
    dt  = float(d["dt"]); Rm = float(d["R"]); warm = float(d["warm"])
    ref = float(d["ref_drms"])
    Phi = np.asarray(d["Phi"], dtype=float)
    if Phi.shape[0] == nx:
        Phi = np.moveaxis(Phi, 2, 0)
    else:
        Phi = Phi.transpose(0, 2, 1)        # undo h5py transpose
    A = np.asarray(d["A"], dtype=float)
    if A.shape[0] != N: A = A.T
    meas = np.asarray(d["meas"], dtype=float).ravel()
    ins_lat = np.asarray(d["ins_lat"]).ravel(); ins_lon = np.asarray(d["ins_lon"]).ravel()
    true_lat = np.asarray(d["true_lat"]).ravel(); true_lon = np.asarray(d["true_lon"]).ravel()
    P0 = np.asarray(d["P0"], dtype=float); Qd = np.asarray(d["Qd"], dtype=float)
    Qd = Qd + 1e-20 * np.eye(nx)            # scale-aware floor (see run_gtsam.py)
    grid = Grid(d["glat"], d["glon"], d["gh"])

    iS  = nx - 1
    iTL = slice(17, 17 + nTL)

    # "previous-line calibration": median converged TL over the last minute of
    # the reference solution (here: same-line batch solution — oracle-flavored,
    # good enough to isolate the cold-start transient)
    est_ref = np.load(npz_path)["est"]
    beta = np.median(est_ref[-600:, iTL], axis=0)
    print(f"warm TL prior |beta| range: {np.abs(beta).min():.3f}..{np.abs(beta).max():.3f}",
          flush=True)

    x0 = np.zeros(nx)
    x0[iTL] = beta
    P0w = P0.copy()
    P0w[iTL, iTL] = np.eye(nTL) * 0.1**2    # calibrated: sigma 1.0 -> 0.1

    prior_nm = gtsam.noiseModel.Gaussian.Covariance(P0w)
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

    params = gtsam.ISAM2Params()
    params.setFactorization("QR")           # conditioning (see run_gtsam.py)
    sm = gtsam_unstable.IncrementalFixedLagSmoother(lag, params)

    KTM = gtsam_unstable.FixedLagSmootherKeyTimestampMap
    graph = gtsam.NonlinearFactorGraph(); vals = gtsam.Values(); ts = KTM()
    est = np.zeros((N, nx)); est_rt = np.zeros((N, nx))
    graph.push_back(gtsam.PriorFactorVector(X(0), x0, prior_nm))
    vals.insert(X(0), x0.copy()); ts.insert((X(0), 0.0))
    graph.add(gtsam.CustomFactor(meas_nm, [X(0)], meas_err(0)))

    t0 = time.time()
    for t in range(1, N):
        graph.add(gtsam.CustomFactor(dyn_nm, [X(t-1), X(t)], dyn_err(Phi[t-1])))
        graph.add(gtsam.CustomFactor(meas_nm, [X(t)], meas_err(t)))
        vals.insert(X(t), x0.copy()); ts.insert((X(t), t * dt))
        sm.update(graph, vals, ts)
        graph = gtsam.NonlinearFactorGraph(); vals = gtsam.Values(); ts = KTM()
        cur = sm.calculateEstimate()
        est_rt[t] = cur.atVector(X(t))
        for k in range(max(0, t - int(round(lag/dt))), t + 1):
            if cur.exists(X(k)):
                est[k] = cur.atVector(X(k))
        if t % 1000 == 0:
            el = time.time() - t0
            print(f"t={t}/{N}  elapsed={el:.0f}s", flush=True)

    est_lat = ins_lat + est[:, 0]; est_lon = ins_lon + est[:, 1]
    rt_lat  = ins_lat + est_rt[:, 0]; rt_lon = ins_lon + est_rt[:, 1]
    np.savez(f"research/gtsam_poc/est_warm_lag{lag:g}.npz", est=est, est_rt=est_rt,
             ins_lat=ins_lat, ins_lon=ins_lon, true_lat=true_lat, true_lon=true_lon)

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
        r = drms_of(rt_lat, rt_lon, wv)
        i = drms_of(ins_lat, ins_lon, wv)
        lines.append(f"warm={wv:g}s  smoothed_drms={g:.2f} m  "
                     f"realtime_drms={r:.2f} m  ins_drms={i:.2f} m")
        print(lines[-1], flush=True)
    print(f"(warm-start TL, lag={lag:g}s)", flush=True)
    with open(f"research/gtsam_poc/gtsam_poc_result_warm_lag{lag:g}.txt", "w") as f:
        f.write("\n".join(lines) + f"\nN {N} nx {nx} nTL {nTL}\n")

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
