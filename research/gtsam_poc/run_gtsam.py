#!/usr/bin/env python3
"""GTSAM PoC: reproduce the Julia fixed-lag MagNav FGO on one exported line using
gtsam.IncrementalFixedLagSmoother (ISAM2 + marginalization).

Error-state graph (mirrors src/fgo_online.jl):
  state x_t in R^nx = [dpos(0:2) dvel/datt/bias... , TL beta (17:17+nTL), S (nx-1)]
  prior  : x_0 ~ N(0, P0)
  dynamics: x_{t+1} = Phi_t x_t + w,  w~N(0,Qd)      (linear between-factor)
  measurement: z_t = h(x_t) + v, v~N(0,R), robust Huber
     h(x_t) = mapIGRF(ins_lat_t+dlat, ins_lon_t+dlon) + A_t . beta + S
  The measurement is nonlinear only through mapIGRF(pos), interpolated (bilinear)
  from the exported (map+IGRF) grid; its gradient is a finite difference on the grid.

The fixed-lag smoother marginalizes states older than `lag` (= win), so each
committed (marginalized) state has seen ~win of look-back re-optimization: the
standard incremental sliding-window structure (cf. ASW-FGO / VINS).
"""
import sys, math
import numpy as np
import h5py
import gtsam
import gtsam_unstable                     # IncrementalFixedLagSmoother lives here
from gtsam import symbol_shorthand
X = symbol_shorthand.X

R_EARTH = 6378137.0

def load(path):
    d = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            d[k] = f[k][()]
    # HDF5/Julia stores column-major; h5py returns arrays with reversed axes.
    # Normalize the ones we index explicitly.
    return d

class Grid:
    """Bilinear (map+IGRF) grid on (lat,lon) axes, with finite-difference gradient."""
    def __init__(self, glat, glon, gh):
        self.glat = np.asarray(glat).ravel()
        self.glon = np.asarray(glon).ravel()
        gh = np.asarray(gh)
        # ensure gh[i,j] corresponds to (glat[i], glon[j])
        if gh.shape != (self.glat.size, self.glon.size):
            gh = gh.T
        self.gh = gh
        self.dlat = self.glat[1] - self.glat[0]
        self.dlon = self.glon[1] - self.glon[0]

    def _bilin(self, lat, lon):
        i = np.clip((lat - self.glat[0]) / self.dlat, 0, self.glat.size - 1.001)
        j = np.clip((lon - self.glon[0]) / self.dlon, 0, self.glon.size - 1.001)
        i0, j0 = int(i), int(j); fi, fj = i - i0, j - j0
        g = self.gh
        v = (g[i0, j0]*(1-fi)*(1-fj) + g[i0+1, j0]*fi*(1-fj)
             + g[i0, j0+1]*(1-fi)*fj + g[i0+1, j0+1]*fi*fj)
        return v

    def value(self, lat, lon):
        return self._bilin(lat, lon)

    def grad(self, lat, lon):
        h = 1e-6  # rad
        dvlat = (self._bilin(lat+h, lon) - self._bilin(lat-h, lon)) / (2*h)
        dvlon = (self._bilin(lat, lon+h) - self._bilin(lat, lon-h)) / (2*h)
        return dvlat, dvlon


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "research/gtsam_poc/line_1007_06.h5"
    d = load(path)
    N   = int(d["N"]); nx = int(d["nx"]); nTL = int(d["nTL"])
    dt  = float(d["dt"]); Rm = float(d["R"]); warm = float(d["warm"])
    win = float(d["win"]); ref = float(d["ref_drms"])
    Phi = np.asarray(d["Phi"], dtype=float)     # want [N-1, nx, nx]
    if Phi.shape[0] == nx:      # stored [nx,nx,N-1] -> transpose to [N-1,nx,nx]
        Phi = np.moveaxis(Phi, 2, 0)
    A   = np.asarray(d["A"], dtype=float)
    if A.shape[0] != N: A = A.T                  # -> [N, nTL]
    meas = np.asarray(d["meas"], dtype=float).ravel()
    ins_lat = np.asarray(d["ins_lat"]).ravel(); ins_lon = np.asarray(d["ins_lon"]).ravel()
    true_lat = np.asarray(d["true_lat"]).ravel(); true_lon = np.asarray(d["true_lon"]).ravel()
    P0 = np.asarray(d["P0"], dtype=float); Qd = np.asarray(d["Qd"], dtype=float)
    # the Pinson position process noise is ~0 (position error propagates
    # deterministically from velocity), so Covariance(Qd) is near-singular; floor
    # the diagonal so the dynamics factor is representable (cf. fgo_gn_step q_floor).
    Qd = Qd + 1e-8 * np.eye(nx)
    grid = Grid(d["glat"], d["glon"], d["gh"])

    iS  = nx - 1                                 # S state index (last)
    iTL = slice(17, 17 + nTL)                    # TL beta indices (0-based)

    prior_nm = gtsam.noiseModel.Gaussian.Covariance(P0)
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
    sm = gtsam_unstable.IncrementalFixedLagSmoother(win, params)

    KTM = gtsam_unstable.FixedLagSmootherKeyTimestampMap
    graph = gtsam.NonlinearFactorGraph(); vals = gtsam.Values(); ts = KTM()
    est = np.zeros((N, nx))
    graph.push_back(gtsam.PriorFactorVector(X(0), np.zeros(nx), prior_nm))
    vals.insert(X(0), np.zeros(nx)); ts.insert((X(0), 0.0))
    graph.add(gtsam.CustomFactor(meas_nm, [X(0)], meas_err(0)))

    for t in range(1, N):
        graph.add(gtsam.CustomFactor(dyn_nm, [X(t-1), X(t)], dyn_err(Phi[t-1])))
        graph.add(gtsam.CustomFactor(meas_nm, [X(t)], meas_err(t)))
        vals.insert(X(t), np.zeros(nx)); ts.insert((X(t), t * dt))
        sm.update(graph, vals, ts)
        graph = gtsam.NonlinearFactorGraph(); vals = gtsam.Values(); ts = KTM()
        cur = sm.calculateEstimate()
        # record the freshest available estimate for each key still in the window
        for k in range(max(0, t - int(round(win/dt))), t + 1):
            if cur.exists(X(k)):
                est[k] = cur.atVector(X(k))

    est_lat = ins_lat + est[:, 0]; est_lon = ins_lon + est[:, 1]
    m = (np.arange(N) * dt) >= warm
    dn = (est_lat[m] - true_lat[m]) * R_EARTH
    de = (est_lon[m] - true_lon[m]) * R_EARTH * np.cos(true_lat[m])
    drms = math.sqrt(np.mean(dn**2 + de**2))
    print(f"GTSAM fixed-lag DRMS = {drms:.2f} m   (Julia FGO reference = {ref:.2f} m)")
    with open("research/gtsam_poc/gtsam_poc_result.txt", "w") as f:
        f.write(f"gtsam_drms {drms:.3f}\njulia_ref {ref:.3f}\nN {N} nx {nx} nTL {nTL}\n")

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        with open("research/gtsam_poc/gtsam_poc_result.txt", "w") as f:
            f.write("PYTHON_ERROR\n" + tb)
        raise
