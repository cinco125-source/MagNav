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
        # Julia writes gh[i=lat, j=lon]; h5py's column-major reversal hands us
        # the transpose, for ANY shape — including the square 200x200 grid the
        # old shape-check silently passed through (verified: residual std vs
        # meas is 31 nT transposed, 242 nT as-is). Always undo it.
        self.gh = np.asarray(gh).T
        assert self.gh.shape == (self.glat.size, self.glon.size)
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
    # optional CLI override of the smoother lag [s] (default: exported win=300)
    if len(sys.argv) > 2:
        win = float(sys.argv[2])
    tag = f"_lag{win:g}" if len(sys.argv) > 2 else ""
    Phi = np.asarray(d["Phi"], dtype=float)     # want [N-1, nx, nx]
    if Phi.shape[0] == nx:      # stored [nx,nx,N-1] -> transpose to [N-1,nx,nx]
        Phi = np.moveaxis(Phi, 2, 0)
    else:
        # h5py reads Julia's column-major [nx,nx,N-1] as [N-1,nx,nx] with each
        # slice TRANSPOSED (verified: dt/R shows up at [3,0] instead of [0,3]).
        Phi = Phi.transpose(0, 2, 1)
    A   = np.asarray(d["A"], dtype=float)
    if A.shape[0] != N: A = A.T                  # -> [N, nTL]
    meas = np.asarray(d["meas"], dtype=float).ravel()
    ins_lat = np.asarray(d["ins_lat"]).ravel(); ins_lon = np.asarray(d["ins_lon"]).ravel()
    true_lat = np.asarray(d["true_lat"]).ravel(); true_lon = np.asarray(d["true_lon"]).ravel()
    P0 = np.asarray(d["P0"], dtype=float); Qd = np.asarray(d["Qd"], dtype=float)
    # the Pinson position process noise is ~0 (diag 1e-31 rad^2: position error
    # propagates deterministically from velocity), so whitening Covariance(Qd)
    # spans ~30 orders of magnitude and elimination can go indeterminate (x226).
    # Floor at 1e-20: for lat/lon that is sigma ~0.6 mm/step (5 cm random walk
    # over the whole 600 s segment, negligible), and caps whitened weights at
    # ~1e10. A 1e-8 floor would be sigma ~640 m/step in rad states — it destroys
    # the dynamics and the estimate collapses onto the INS.
    Qd = Qd + 1e-20 * np.eye(nx)
    grid = Grid(d["glat"], d["glon"], d["gh"])

    # data sanity: a NaN/Inf in Phi/meas/INS shows up as an ISAM2 "indeterminate"
    for nm_, arr in (("Phi", Phi), ("A", A), ("meas", meas),
                     ("ins_lat", ins_lat), ("ins_lon", ins_lon)):
        bad = ~np.isfinite(arr)
        if bad.any():
            if arr.ndim == 1:
                epochs = np.where(bad)[0][:10].tolist()
            else:
                epochs = sorted(set(np.where(bad.any(axis=tuple(range(1, arr.ndim)))
                                             if arr.ndim > 1 else bad)[0].tolist()))[:10]
            msg = f"NONFINITE in {nm_}: {int(bad.sum())} entries, first epochs {epochs}\n"
            print(msg)
            with open("research/gtsam_poc/gtsam_poc_result.txt", "w") as f:
                f.write("DATA_NONFINITE\n" + msg)

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
    # Cholesky squares the conditioning: with Qd floored at 1e-20 the info
    # diagonal spans ~1e20 and elimination goes indeterminate at t=1. QR works
    # on the whitened Jacobian (kappa ~1e10), which double precision handles.
    params.setFactorization("QR")
    sm = gtsam_unstable.IncrementalFixedLagSmoother(win, params)
    # NO per-variable reg prior: P0*1e4 gives a per-epoch pull to dpos=0 of
    # sigma ~10 m; 6000 of them aggregate to sigma ~0.13 m, which pins the
    # whole trajectory onto the INS (measured: gtsam DRMS == INS DRMS to cm).
    # With the x0 prior and a nonsingular Qd the chain is fully constrained.

    KTM = gtsam_unstable.FixedLagSmootherKeyTimestampMap
    graph = gtsam.NonlinearFactorGraph(); vals = gtsam.Values(); ts = KTM()
    est = np.zeros((N, nx))       # smoothed: value when the state leaves the lag window
    est_rt = np.zeros((N, nx))    # real-time: value of the NEWEST state right after update
    graph.push_back(gtsam.PriorFactorVector(X(0), np.zeros(nx), prior_nm))
    vals.insert(X(0), np.zeros(nx)); ts.insert((X(0), 0.0))
    graph.add(gtsam.CustomFactor(meas_nm, [X(0)], meas_err(0)))

    import time
    t0 = time.time()
    for t in range(1, N):
        graph.add(gtsam.CustomFactor(dyn_nm, [X(t-1), X(t)], dyn_err(Phi[t-1])))
        graph.add(gtsam.CustomFactor(meas_nm, [X(t)], meas_err(t)))
        vals.insert(X(t), np.zeros(nx)); ts.insert((X(t), t * dt))
        try:
            sm.update(graph, vals, ts)
        except RuntimeError:
            print(f"FAILED at t={t}", flush=True)
            raise
        graph = gtsam.NonlinearFactorGraph(); vals = gtsam.Values(); ts = KTM()
        cur = sm.calculateEstimate()
        est_rt[t] = cur.atVector(X(t))   # causal estimate available at time t
        # record the freshest available estimate for each key still in the window
        for k in range(max(0, t - int(round(win/dt))), t + 1):
            if cur.exists(X(k)):
                est[k] = cur.atVector(X(k))
        if t % 100 == 0:
            el = time.time() - t0
            print(f"t={t}/{N}  elapsed={el:.0f}s  ({el/t*1000:.0f} ms/epoch)", flush=True)

    est_lat = ins_lat + est[:, 0]; est_lon = ins_lon + est[:, 1]
    rt_lat  = ins_lat + est_rt[:, 0]; rt_lon = ins_lon + est_rt[:, 1]
    np.savez(f"research/gtsam_poc/est_gtsam{tag}.npz", est=est, est_rt=est_rt,
             est_lat=est_lat, est_lon=est_lon, rt_lat=rt_lat, rt_lon=rt_lon,
             ins_lat=ins_lat, ins_lon=ins_lon,
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
        r = drms_of(rt_lat, rt_lon, wv)
        i = drms_of(ins_lat, ins_lon, wv)
        lines.append(f"warm={wv:g}s  smoothed_drms={g:.2f} m  "
                     f"realtime_drms={r:.2f} m  ins_drms={i:.2f} m")
        print(lines[-1], flush=True)
    print(f"(Julia FGO reference = {ref:.2f} m, exported warm={warm:g}s, lag={win:g}s)",
          flush=True)
    with open(f"research/gtsam_poc/gtsam_poc_result{tag}.txt", "w") as f:
        f.write("\n".join(lines) + f"\njulia_ref {ref:.3f}\nN {N} nx {nx} nTL {nTL}\n")

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
