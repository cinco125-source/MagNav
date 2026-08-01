#!/usr/bin/env python3
"""Per-epoch incremental smoother with the compensation carried as SEPARATE,
long-lived graph variables instead of a per-epoch block of the navigation state.

Motivation (research/gtsam_poc/knee_ridge.log). Solving the first W seconds of
line 1003.02 Mag 4 as a converged batch gives DRMS 2135 m at W=100 s against
42 m at W=200 s -- and the 100 s solve reaches the LOWER objective (1.41e4 vs
1.56e4). The short-window joint problem is therefore not badly optimized, it is
weakly identifiable: a position error together with compensating Tolles-Lawson
coefficients explains the scalar measurement at least as well as the truth, and
|beta|max runs to 546 doing it. A fixed-lag smoother that commits and
marginalizes on that window freezes the wrong branch permanently, which is the
one way it can end up behind a causal EKF that never had the freedom to find it.

In run_gtsam_decimated.py the coefficients live inside the per-epoch state
x_t (indices 17:17+nTL), so a 300 s lag carries 300 independent 19-vectors of
compensation, and each one is marginalized 300 s after its epoch -- the whole
flight's information about a nominally slowly-varying quantity is repeatedly
compressed into a boundary prior and then diluted by the random walk. Here the
graph is split:

  Y(s) in R^18  navigation error: Pinson states 0..16 plus the FOGM state S
                (original indices 0..16 and nx-1), placed every K raw samples
                and marginalized on the fixed lag exactly as before
  B(j) in R^nTL Tolles-Lawson coefficients, one node per --beta-period seconds,
                chained by a random-walk factor and (by default) re-stamped to
                the current time at every update so they are never marginalized

The measurement factor becomes binary on (Y(s), B(j(s))). Nothing else changes:
same error model, same measurement model, same noise settings, same Huber
kernel, same decimation, same lag.

The split is exact, not an approximation. get_pinson builds F = zeros(nx,nx) and
writes only the 17 inertial rows/columns plus F[end,end] = -1/fogm_tau, so the
Tolles-Lawson rows and columns are identically zero: Phi = exp(F dt) is block
diagonal with an identity TL block, and create_P0 / create_Qd are block diagonal
by construction. Slicing Phi, Qd and P0 on the two index sets loses nothing.

What this is meant to buy: over a 300 s lag the compensation has nTL degrees of
freedom instead of 300*nTL, and it keeps accumulating information from the whole
flight rather than being frozen at a cold-start linearization. If the diagnosis
above is right, the cases where the smoother currently trails the EKF
(1007.02 Mag 5, 1003.08 Mag 4) are the ones that should move.

CONTROL RUN FIRST. `--beta-period 1 --no-beta-persist` puts one B node per state
and lets it age out on the lag, which is the existing per-epoch scheme expressed
in the split parameterization. It should reproduce run_gtsam_decimated.py to
within the elimination-order difference. Run it before trusting any other
number from this script.

Usage: run_gtsam_split.py <line.h5> [lag_s=300] [mag 4|5] [K=10] [tag]
                          [--beta-period T] [--no-beta-persist]
                          [--tl-sigma S] [--sigma-s S] [--tl-walk W]
                          [--norobust | --huber C] [--qfloor F] [--rsigma X]
                          [--tl-init T] [--full-relin] [--dense-meas]
--beta-period T: seconds of flight per compensation node (default 600).
                 T=0 places a single static node for the whole line.
--no-beta-persist: let compensation nodes age out of the lag window like any
                 other variable, instead of holding them in the graph.
"""
import sys, math, time
import numpy as np
import h5py
import gtsam
import gtsam_unstable
from gtsam import symbol_shorthand
Y = symbol_shorthand.X          # navigation error states
B = symbol_shorthand.B          # Tolles-Lawson compensation nodes

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


def argf(name, default):
    return float(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def main():
    path = sys.argv[1]
    lag = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    mag = sys.argv[3] if len(sys.argv) > 3 else "5"
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    tag = sys.argv[5] if len(sys.argv) > 5 and not sys.argv[5].startswith("--") \
        else f"_split_lag{lag:g}_m{mag}"
    T_beta = argf("--beta-period", 600.0)
    persist = "--no-beta-persist" not in sys.argv

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
    if "--rsigma" in sys.argv:
        Rm = argf("--rsigma", 12.0) ** 2
        print(f"measurement sigma set to {math.sqrt(Rm):g} nT", flush=True)
    qf = argf("--qfloor", 1e-20)
    Qd = Qd + qf * np.eye(nx)
    grid = Grid(d["glat"], d["glon"], d["gh_rowmajor"])

    # index sets. The original state is [Pinson 0..16 | TL 17..17+nTL-1 | S nx-1].
    iS_full = nx - 1
    tl_idx = np.arange(17, 17 + nTL)
    nav_idx = np.concatenate([np.arange(0, 17), [iS_full]])
    nY = nav_idx.size                                # 18
    iSy = nY - 1                                     # S sits last inside Y
    assert nY + nTL == nx, f"index split {nY}+{nTL} != {nx}"

    # the split is only exact if the cross blocks vanish; check rather than assume
    for nm_, Mx in (("P0", P0), ("Qd", Qd), ("Phi[0]", Phi[0])):
        x = np.abs(Mx[np.ix_(nav_idx, tl_idx)]).max()
        y = np.abs(Mx[np.ix_(tl_idx, nav_idx)]).max()
        assert max(x, y) < 1e-12, f"{nm_} couples navigation and TL blocks ({x:.3g},{y:.3g})"
    assert np.abs(Phi[0][np.ix_(tl_idx, tl_idx)] - np.eye(nTL)).max() < 1e-12, \
        "TL transition is not identity"

    # decimated chain: keep samples 0, K, 2K, ...; compound Phi/Qd in between
    idx = np.arange(0, N, K)
    M = idx.size
    dtK = dt * K
    PhiK = np.zeros((M-1, nx, nx)); QdK = np.zeros((M-1, nx, nx))
    for s in range(M-1):
        P_ = np.eye(nx); Q_ = np.zeros((nx, nx))
        for t in range(idx[s], min(idx[s+1], N-1)):
            P_ = Phi[t] @ P_
            Q_ = Phi[t] @ Q_ @ Phi[t].T + Qd
        PhiK[s] = P_; QdK[s] = (Q_ + Q_.T) / 2

    sig_TL = 1.0
    if "--tl-sigma" in sys.argv:
        sig_TL = argf("--tl-sigma", 1.0)
        P0 = P0.copy(); P0[np.ix_(tl_idx, tl_idx)] *= sig_TL ** 2
        print(f"TL prior sigma scaled by {sig_TL:g}", flush=True)
    if "--tl-cols" in sys.argv:
        Sc = argf("--tl-cols", 100.0)
        sd = Sc / np.maximum(np.sqrt((A ** 2).mean(axis=0)), 1e-12)
        P0 = P0.copy(); P0[np.ix_(tl_idx, tl_idx)] = np.diag(sd ** 2)
        print(f"TL prior per column: contribution sigma {Sc:g} nT", flush=True)
    if "--sigma-s" in sys.argv:
        Ss = argf("--sigma-s", 1.0)
        P0 = P0.copy(); P0[iS_full, iS_full] *= Ss ** 2
        for s in range(M - 1):
            QdK[s][iS_full, iS_full] *= Ss ** 2
        print(f"FOGM prior and walk sigma scaled by {Ss:g}", flush=True)
    if "--tl-walk" in sys.argv:
        Wk = argf("--tl-walk", 1.0)
        for s in range(M - 1):
            QdK[s][np.ix_(tl_idx, tl_idx)] *= Wk ** 2
        print(f"TL random walk sigma scaled by {Wk:g}", flush=True)

    x0_TL = np.zeros(nTL)
    if "--tl-init" in sys.argv:
        T_init = argf("--tl-init", 300.0)
        n_init = min(N, int(round(T_init / dt)))
        r0 = np.array([meas[t] - grid.value(ins_lat[t], ins_lon[t])
                       for t in range(n_init)])
        Aw = A[:n_init]
        lam = Rm / sig_TL ** 2
        x0_TL = np.linalg.solve(Aw.T @ Aw + lam * np.eye(nTL), Aw.T @ r0)
        print(f"TL bootstrap over first {T_init:g}s: |beta|max={np.abs(x0_TL).max():.1f}",
              flush=True)

    # split the model matrices
    PhiY = PhiK[:, nav_idx][:, :, nav_idx]
    QdY = QdK[:, nav_idx][:, :, nav_idx]
    P0Y = P0[np.ix_(nav_idx, nav_idx)]
    P0B = P0[np.ix_(tl_idx, tl_idx)]
    QB_step = QdK[:, tl_idx][:, :, tl_idx]           # per 1 s, TL block

    # compensation node schedule. beta_steps states share one node; Phi_TL = I so
    # the random walk over an interval is the sum of the per-step covariances.
    if T_beta <= 0:
        beta_steps = M                               # one static node for the line
    else:
        beta_steps = max(1, int(round(T_beta / dtK)))
    n_beta = max(1, int(math.ceil(M / beta_steps)))
    bnode = np.minimum(np.arange(M) // beta_steps, n_beta - 1)
    print(f"lag={lag:g}s K={K} mag={mag} ({meas_key}) N={N} -> {M} nav states, "
          f"{n_beta} compensation nodes ({beta_steps*dtK:g}s each), "
          f"persist={persist}", flush=True)

    prior_nmY = gtsam.noiseModel.Gaussian.Covariance(P0Y)
    prior_nmB = gtsam.noiseModel.Gaussian.Covariance(P0B)
    meas_base = gtsam.noiseModel.Isotropic.Sigma(1, math.sqrt(Rm))
    if "--norobust" in sys.argv:
        meas_nm = meas_base
        print("no robust kernel (pure Gaussian measurement factors)", flush=True)
    else:
        hub_c = argf("--huber", 1.345)
        meas_nm = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(hub_c), meas_base)

    def dyn_err(Phi_t):
        def f(this, values, J):
            xa = values.atVector(this.keys()[0]); xb = values.atVector(this.keys()[1])
            e = xb - Phi_t @ xa
            if J is not None:
                J[0] = -Phi_t; J[1] = np.eye(nY)
            return e
        return f

    def walk_err(this, values, J):
        ba = values.atVector(this.keys()[0]); bb = values.atVector(this.keys()[1])
        e = bb - ba
        if J is not None:
            J[0] = -np.eye(nTL); J[1] = np.eye(nTL)
        return e

    def meas_err(t):
        """Binary factor on (Y(s), B(j)): z = map(ins+dpos) + A_t.beta + S + v."""
        lat0, lon0, At, zt = ins_lat[t], ins_lon[t], A[t], meas[t]
        def f(this, values, J):
            y = values.atVector(this.keys()[0]); b = values.atVector(this.keys()[1])
            lat, lon = lat0 + y[0], lon0 + y[1]
            h = grid.value(lat, lon) + At @ b + y[iSy]
            e = np.array([h - zt])
            if J is not None:
                jy = np.zeros((1, nY))
                glat_, glon_ = grid.grad(lat, lon)
                jy[0, 0] = glat_; jy[0, 1] = glon_; jy[0, iSy] = 1.0
                J[0] = jy
                J[1] = At.reshape(1, nTL)
            return e
        return f

    params = gtsam.ISAM2Params()
    params.setFactorization("QR")
    if "--full-relin" in sys.argv:
        params.setRelinearizeThreshold(0.0)
        params.relinearizeSkip = 1
        print("full relinearization every update", flush=True)
    sm = gtsam_unstable.IncrementalFixedLagSmoother(lag, params)
    KTM = gtsam_unstable.FixedLagSmootherKeyTimestampMap

    dense = "--dense-meas" in sys.argv

    def add_meas(graph, s):
        rng = range(idx[s], min(idx[s] + K, N)) if dense else (idx[s],)
        for t in rng:
            graph.add(gtsam.CustomFactor(meas_nm, [Y(s), B(int(bnode[s]))], meas_err(t)))

    def stamp(ts, s):
        """Timestamps for one update. Compensation nodes are re-stamped to the
        current time when persisting, which is what keeps them out of the
        smoother's marginalization sweep; navigation states age normally."""
        t_now = s * dtK
        if persist:
            for j in range(int(bnode[s]) + 1):
                ts.insert((B(j), t_now))
        else:
            ts.insert((B(int(bnode[s])), t_now))

    est = np.zeros((M, nY)); est_rt = np.zeros((M, nY))
    beta_rt = np.zeros((M, nTL))
    graph = gtsam.NonlinearFactorGraph(); vals = gtsam.Values(); ts = KTM()
    graph.push_back(gtsam.PriorFactorVector(Y(0), np.zeros(nY), prior_nmY))
    graph.push_back(gtsam.PriorFactorVector(B(0), x0_TL, prior_nmB))
    vals.insert(Y(0), np.zeros(nY)); vals.insert(B(0), x0_TL)
    ts.insert((Y(0), 0.0)); stamp(ts, 0)
    add_meas(graph, 0)
    sm.update(graph, vals, ts)
    cur = sm.calculateEstimate()
    est_rt[0] = cur.atVector(Y(0)); beta_rt[0] = cur.atVector(B(0))
    graph = gtsam.NonlinearFactorGraph(); vals = gtsam.Values(); ts = KTM()

    t0 = time.time()
    lag_states = int(round(lag / dtK))
    for s in range(1, M):
        graph.add(gtsam.CustomFactor(gtsam.noiseModel.Gaussian.Covariance(QdY[s-1]),
                                     [Y(s-1), Y(s)], dyn_err(PhiY[s-1])))
        j, jp = int(bnode[s]), int(bnode[s-1])
        if j != jp:
            # a new compensation node opens: chain it to its predecessor with the
            # random walk accumulated over the interval the predecessor covered
            QB = QB_step[max(0, s-beta_steps):s].sum(axis=0)
            QB = (QB + QB.T) / 2 + 1e-18 * np.eye(nTL)
            graph.add(gtsam.CustomFactor(gtsam.noiseModel.Gaussian.Covariance(QB),
                                         [B(jp), B(j)], walk_err))
            vals.insert(B(j), cur.atVector(B(jp)))
        add_meas(graph, s)
        x_init = PhiY[s-1] @ est_rt[s-1] if s > 1 else np.zeros(nY)
        vals.insert(Y(s), x_init)
        ts.insert((Y(s), s * dtK)); stamp(ts, s)
        try:
            sm.update(graph, vals, ts)
        except RuntimeError:
            print(f"FAILED at state {s} (t={idx[s]*dt:.0f}s)", flush=True)
            raise
        graph = gtsam.NonlinearFactorGraph(); vals = gtsam.Values(); ts = KTM()
        cur = sm.calculateEstimate()
        est_rt[s] = cur.atVector(Y(s))
        beta_rt[s] = cur.atVector(B(j)) if cur.exists(B(j)) else beta_rt[s-1]
        for k in range(max(0, s - lag_states), s + 1):
            if cur.exists(Y(k)):
                est[k] = cur.atVector(Y(k))
        if s % 500 == 0:
            el = time.time() - t0
            print(f"s={s}/{M}  elapsed={el:.0f}s  ({el/s*1000:.1f} ms/state)", flush=True)

    il_ = ins_lat[idx]; io_ = ins_lon[idx]
    tl_ = true_lat[idx]; to_ = true_lon[idx]
    est_lat = il_ + est[:, 0]; est_lon = io_ + est[:, 1]
    rt_lat = il_ + est_rt[:, 0]; rt_lon = io_ + est_rt[:, 1]
    np.savez(f"research/gtsam_poc/est_gtsam{tag}.npz", est=est, est_rt=est_rt,
             beta_rt=beta_rt, est_lat=est_lat, est_lon=est_lon,
             rt_lat=rt_lat, rt_lon=rt_lon, ins_lat=il_, ins_lon=io_,
             true_lat=tl_, true_lon=to_)

    def drms_of(lat_, lon_, warm_v):
        m = (idx * dt) >= warm_v
        if not m.any():
            return float("nan")
        dn = (lat_[m] - tl_[m]) * R_EARTH
        de = (lon_[m] - to_[m]) * R_EARTH * np.cos(tl_[m])
        return math.sqrt(np.mean(dn**2 + de**2))

    lines = []
    for wv in (0.0, 60.0, 120.0, 300.0, warm):
        lines.append(f"warm={wv:g}s  smoothed_drms={drms_of(est_lat,est_lon,wv):.2f} m  "
                     f"realtime_drms={drms_of(rt_lat,rt_lon,wv):.2f} m  "
                     f"ins_drms={drms_of(il_,io_,wv):.2f} m")
        print(lines[-1], flush=True)
    el = time.time() - t0
    with open(f"research/gtsam_poc/gtsam_poc_result{tag}.txt", "w") as f:
        f.write("\n".join(lines) +
                f"\nlag {lag:g} K {K} mag {mag} beta_period {T_beta:g} "
                f"n_beta {n_beta} persist {int(persist)} "
                f"ms_per_state {1e3*el/max(1,M-1):.1f}\n")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        with open("research/gtsam_poc/gtsam_poc_result_split_error.txt", "w") as f:
            f.write("PYTHON_ERROR\n" + tb)
        raise
