#!/usr/bin/env python3
"""Full-batch LM variant of run_gtsam.py: same factors (prior, linear dynamics,
robust map-matching measurement, diffuse reg prior), one LM solve over all N
states instead of the incremental fixed-lag smoother. Gives a fast DRMS number
(offline smoother bound) while the fixed-lag run completes.
"""
import sys, math
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
        # undo h5py's transpose of Julia's column-major grid (see run_gtsam_prog.py)
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
    ref = float(d["ref_drms"])
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
    P0 = np.asarray(d["P0"], dtype=float); Qd = np.asarray(d["Qd"], dtype=float)
    # scale-aware floor: position Qd diag is 1e-31 rad^2; 1e-20 is ~0.6 mm/step
    # (negligible) but keeps whitened weights <= ~1e10. See run_gtsam_prog.py.
    Qd = Qd + 1e-20 * np.eye(nx)
    grid = Grid(d["glat"], d["glon"], d["gh"])

    iS  = nx - 1
    iTL = slice(17, 17 + nTL)

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

    graph = gtsam.NonlinearFactorGraph()
    vals = gtsam.Values()
    graph.push_back(gtsam.PriorFactorVector(X(0), np.zeros(nx), prior_nm))
    for t in range(N):
        graph.add(gtsam.CustomFactor(meas_nm, [X(t)], meas_err(t)))
        vals.insert(X(t), np.zeros(nx))
        if t >= 1:
            graph.add(gtsam.CustomFactor(dyn_nm, [X(t-1), X(t)], dyn_err(Phi[t-1])))

    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    params.setMaxIterations(30)
    # default multifrontal Cholesky squares the ~1e10 whitened conditioning and
    # the LM step collapses (gives up at iter 1); QR stays within double range
    params.setLinearSolverType("MULTIFRONTAL_QR")
    opt = gtsam.LevenbergMarquardtOptimizer(graph, vals, params)
    sol = opt.optimize()
    print(f"LM iters={opt.iterations()}  error={opt.error():.3f}", flush=True)

    est = np.zeros((N, nx))
    for t in range(N):
        est[t] = sol.atVector(X(t))

    est_lat = ins_lat + est[:, 0]; est_lon = ins_lon + est[:, 1]
    np.savez("research/gtsam_poc/est_batch.npz", est=est, est_lat=est_lat,
             est_lon=est_lon, ins_lat=ins_lat, ins_lon=ins_lon,
             true_lat=true_lat, true_lon=true_lon)

    # NOTE: the exported warm (600 s) equals the segment length (N*dt = 600 s),
    # so the shipped mask is empty and ref_drms is NaN. Report a warm sweep.
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
        lines.append(f"warm={wv:g}s  gtsam_batch_drms={g:.2f} m  ins_drms={i:.2f} m")
        print(lines[-1], flush=True)
    print(f"(Julia FGO reference = {ref:.2f} m, exported warm={warm:g}s)", flush=True)
    with open("research/gtsam_poc/gtsam_poc_result_batch.txt", "w") as f:
        f.write("\n".join(lines) + f"\njulia_ref {ref:.3f}\nN {N} nx {nx} nTL {nTL}\n")

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        with open("research/gtsam_poc/gtsam_poc_result_batch.txt", "w") as f:
            f.write("PYTHON_ERROR\n" + tb)
        raise
