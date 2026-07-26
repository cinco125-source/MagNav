#!/usr/bin/env python3
"""Per-epoch incremental smoother at a decimated state cadence.

Keeps the fixed-lag LOOK-AHEAD TIME (e.g. 300 s) while placing graph states
every K raw samples (default K=10, i.e. 1 Hz for the 10 Hz SGL data):
  - Phi is compounded exactly over K steps:  Phi' = Phi_{t+K-1} ... Phi_t
  - Qd is propagated exactly:                Q' <- Phi Q' Phi^T + Qd  (K times)
  - the measurement at each kept sample is attached to its state (the
    intermediate K-1 measurements are dropped: map information is spatially
    correlated, so the loss is small; this is the honest cost of the speedup)
This makes lag=300 s a 300-state window updated once per second: real-time
with wide margin, and a full line runs in minutes instead of days.

Usage: run_gtsam_decimated.py <line.h5> [lag_s=300] [mag 4|5] [K=10] [tag]
                              [--full-relin]
--full-relin: relinearize every variable at every update (threshold 0, skip 1),
making each window pass equivalent to a fresh batch iteration; tests whether
the Mag-4 divergence is caused by ISAM2's partial relinearization baking in
early cold-start errors before marginalization makes them permanent.
--dense-meas: attach ALL K raw measurements to their nearest kept state (each
with its own INS reference and TL row; the error state varies slowly over the
K-sample span, so the approximation is mild). Tests whether the Mag-4
divergence is caused by the 1-in-K measurement thinning.
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
    def __init__(self, glat, glon, gh_rowmajor):
        self.glat = np.asarray(glat).ravel()
        self.glon = np.asarray(glon).ravel()
        self.gh = np.asarray(gh_rowmajor)
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
    lag = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    mag = sys.argv[3] if len(sys.argv) > 3 else "5"
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    tag = sys.argv[5] if len(sys.argv) > 5 else f"_dec{K}_lag{lag:g}_m{mag}"

    d = load(path)
    N = int(d["N"]); nx = int(d["nx"]); nTL = int(d["nTL"])
    dt = float(d["dt"]); Rm = float(d["R"]); warm = float(d["warm"])
    Phi = np.asarray(d["Phi"], dtype=float)          # [N-1, nx, nx] (py export)
    A = np.asarray(d["A"], dtype=float)
    meas_key = f"meas_mag{mag}" if f"meas_mag{mag}" in d else "meas"
    meas = np.asarray(d[meas_key], dtype=float).ravel()
    ins_lat = np.asarray(d["ins_lat"]).ravel(); ins_lon = np.asarray(d["ins_lon"]).ravel()
    true_lat = np.asarray(d["true_lat"]).ravel(); true_lon = np.asarray(d["true_lon"]).ravel()
    P0 = np.asarray(d["P0"], dtype=float); Qd = np.asarray(d["Qd"], dtype=float)
    Qd = Qd + 1e-20 * np.eye(nx)                     # scale-aware floor (run_gtsam.py)
    grid = Grid(d["glat"], d["glon"], d["gh_rowmajor"])
    print(f"lag={lag:g}s K={K} mag={mag} ({meas_key}) N={N} -> {N//K} states",
          flush=True)

    # decimated chain: keep samples 0, K, 2K, ...; compound Phi/Qd in between
    idx = np.arange(0, N, K)
    M = idx.size
    dtK = dt * K
    PhiK = np.zeros((M-1, nx, nx))
    QdK = np.zeros((M-1, nx, nx))
    for s in range(M-1):
        P_ = np.eye(nx); Q_ = np.zeros((nx, nx))
        for t in range(idx[s], min(idx[s+1], N-1)):
            P_ = Phi[t] @ P_
            Q_ = Phi[t] @ Q_ @ Phi[t].T + Qd
        PhiK[s] = P_; QdK[s] = (Q_ + Q_.T) / 2

    iS = nx - 1
    iTL = slice(17, 17 + nTL)

    dyn_nms = [gtsam.noiseModel.Gaussian.Covariance(QdK[s]) for s in range(M-1)]
    prior_nm = gtsam.noiseModel.Gaussian.Covariance(P0)
    meas_base = gtsam.noiseModel.Isotropic.Sigma(1, math.sqrt(Rm))
    meas_nm = gtsam.noiseModel.Robust.Create(
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
    params.setFactorization("QR")                    # conditioning (run_gtsam.py)
    if "--full-relin" in sys.argv:
        params.setRelinearizeThreshold(0.0)
        params.relinearizeSkip = 1
        print("full relinearization every update", flush=True)
    sm = gtsam_unstable.IncrementalFixedLagSmoother(lag, params)
    KTM = gtsam_unstable.FixedLagSmootherKeyTimestampMap

    dense = "--dense-meas" in sys.argv
    if dense:
        print("attaching all raw measurements to nearest kept state", flush=True)

    def add_meas(graph, s):
        if dense:
            for t in range(idx[s], min(idx[s] + K, N)):
                graph.add(gtsam.CustomFactor(meas_nm, [X(s)], meas_err(t)))
        else:
            graph.add(gtsam.CustomFactor(meas_nm, [X(s)], meas_err(idx[s])))

    graph = gtsam.NonlinearFactorGraph(); vals = gtsam.Values(); ts = KTM()
    est = np.zeros((M, nx)); est_rt = np.zeros((M, nx))
    graph.push_back(gtsam.PriorFactorVector(X(0), np.zeros(nx), prior_nm))
    vals.insert(X(0), np.zeros(nx)); ts.insert((X(0), 0.0))
    add_meas(graph, 0)

    t0 = time.time()
    lag_states = int(round(lag / dtK))
    for s in range(1, M):
        graph.add(gtsam.CustomFactor(dyn_nms[s-1], [X(s-1), X(s)], dyn_err(PhiK[s-1])))
        add_meas(graph, s)
        # propagate the previous causal estimate as the new state's initial
        # (= linearization) point, as any filter does. Initializing at zero
        # puts the measurement's linearization at the raw INS position; once
        # the true error is large (cold-start Mag 4) the Huber kernel then
        # downweights the genuine map signal as an outlier and the estimate
        # never recovers.
        x_init = PhiK[s-1] @ est_rt[s-1] if s > 1 else np.zeros(nx)
        vals.insert(X(s), x_init); ts.insert((X(s), s * dtK))
        try:
            sm.update(graph, vals, ts)
        except RuntimeError:
            print(f"FAILED at state {s} (t={idx[s]*dt:.0f}s)", flush=True)
            raise
        graph = gtsam.NonlinearFactorGraph(); vals = gtsam.Values(); ts = KTM()
        cur = sm.calculateEstimate()
        est_rt[s] = cur.atVector(X(s))
        for k in range(max(0, s - lag_states), s + 1):
            if cur.exists(X(k)):
                est[k] = cur.atVector(X(k))
        if s % 500 == 0:
            el = time.time() - t0
            print(f"s={s}/{M}  elapsed={el:.0f}s  ({el/s*1000:.0f} ms/state)",
                  flush=True)

    el = time.time() - t0
    print(f"total {el:.0f}s  ({el/(M-1)*1000:.0f} ms/state, "
          f"{el/(N-1)*1000:.1f} ms/raw-epoch)", flush=True)

    ti = idx * dt
    tl_ = true_lat[idx]; tn_ = true_lon[idx]
    il_ = ins_lat[idx]; io_ = ins_lon[idx]
    est_lat = il_ + est[:, 0]; est_lon = io_ + est[:, 1]
    rt_lat = il_ + est_rt[:, 0]; rt_lon = io_ + est_rt[:, 1]
    np.savez(f"research/gtsam_poc/est_gtsam{tag}.npz", est=est, est_rt=est_rt,
             idx=idx, ins_lat=il_, ins_lon=io_, true_lat=tl_, true_lon=tn_)

    def drms_of(lat_, lon_, warm_v):
        m = ti >= warm_v
        if not m.any():
            return float("nan")
        dn = (lat_[m] - tl_[m]) * R_EARTH
        de = (lon_[m] - tn_[m]) * R_EARTH * np.cos(tl_[m])
        return math.sqrt(np.mean(dn**2 + de**2))

    lines = []
    for wv in (0.0, 60.0, 300.0, warm):
        g = drms_of(est_lat, est_lon, wv)
        r = drms_of(rt_lat, rt_lon, wv)
        i = drms_of(il_, io_, wv)
        lines.append(f"warm={wv:g}s  smoothed_drms={g:.2f} m  "
                     f"realtime_drms={r:.2f} m  ins_drms={i:.2f} m")
        print(lines[-1], flush=True)
    with open(f"research/gtsam_poc/gtsam_poc_result{tag}.txt", "w") as f:
        f.write("\n".join(lines) + f"\nlag {lag:g} K {K} mag {mag} "
                f"ms_per_state {el/(M-1)*1000:.1f}\n")

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
