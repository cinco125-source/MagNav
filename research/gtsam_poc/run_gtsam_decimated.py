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
    # --rsigma X: set the measurement standard deviation to X nT instead of the
    # nominal 12. The post-fit residual on the cabin magnetometers is 47 to
    # 153 nT, so this is the control that asks whether the cold-start divergence
    # is a mis-specified measurement noise rather than a mis-specified
    # compensation prior. --qfloor F replaces the 1e-20 conditioning floor added
    # to the process noise, the other candidate for an accidental fix.
    if "--rsigma" in sys.argv:
        Rm = float(sys.argv[sys.argv.index("--rsigma") + 1]) ** 2
        print(f"measurement sigma set to {math.sqrt(Rm):g} nT", flush=True)
    qf = (float(sys.argv[sys.argv.index("--qfloor") + 1])
          if "--qfloor" in sys.argv else 1e-20)
    Qd = Qd + qf * np.eye(nx)                        # scale-aware floor (run_gtsam.py)
    # --fogm-tau T: the disturbance correlation time, which the export bakes into
    # Phi and Qd at the model's nominal 180 s. Fitting an exponential to the
    # autocorrelation of the residual left by a static 19-dof Tolles-Lawson fit
    # at the true position gives 35 s on line 1007.06 Mag 5, and the measured
    # correlation falls to 0.03 by 60 s where exp(-60/180) is still 0.72. The
    # stationary variance sigma_S^2 lives in P0 and is left alone; for a FOGM
    # Qd = sigma_S^2 * 2 dt / tau, so rescaling by the tau ratio moves the
    # correlation time without touching the amplitude. --sigma-s is the
    # amplitude knob and composes with this one.
    if "--fogm-tau" in sys.argv:
        tau_new = float(sys.argv[sys.argv.index("--fogm-tau") + 1])
        Phi[:, nx-1, nx-1] = np.exp(-dt / tau_new)
        Qd[nx-1, nx-1] = (Qd[nx-1, nx-1] - qf) * (180.0 / tau_new) + qf
        print(f"FOGM tau set to {tau_new:g} s (was 180)", flush=True)
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

    # --tl-init T: bootstrap the Tolles-Lawson coefficients from the first T
    # seconds instead of starting them at zero. diag_window1.py shows the cold
    # start converges to a spurious minimum ~1.4 km from truth on Mag 4 while
    # the same window solved from a ridge-fitted calibration reaches a LOWER
    # objective 57 m from truth; the uncompensated cabin field is 1771 nT rms on
    # Mag 4 against 197 nT on Mag 5, which is why only the cabin magnetometer
    # with the large platform field is affected. The ridge penalty is the TL
    # prior itself (sigma = 1 per coefficient, R on the residual), so this is
    # the maximum a posteriori calibration given the bootstrap data and nothing
    # the smoother does not already have.
    # --tl-sigma S: scale the standard deviation of the Tolles-Lawson block of
    # the initial covariance by S, leaving its mean at zero. The nominal prior is
    # sigma = 1 per coefficient while the cabin field needs coefficients of a few
    # hundred, so this is the control that asks whether the divergence is simply
    # a prior scaled for a compensated installation. --tl-walk W does the same to
    # the TL block of the process noise.
    sig_TL = 1.0                                 # nominal P0 TL block is I
    if "--tl-sigma" in sys.argv:
        sig_TL = float(sys.argv[sys.argv.index("--tl-sigma") + 1])
        P0 = P0.copy(); P0[iTL, iTL] = P0[iTL, iTL] * sig_TL ** 2
        print(f"TL prior sigma scaled by {sig_TL:g}", flush=True)
    # --pos-sigma X: initial horizontal position prior standard deviation in
    # metres (the export was built with 0.1 m). Controls the initial-uncertainty
    # sweep the cold-start claim rests on: at 0.1 m the estimator starts from a
    # known position and unknown compensation; larger values relax the former.
    if "--pos-sigma" in sys.argv:
        ps = float(sys.argv[sys.argv.index("--pos-sigma") + 1])
        sc = (ps / 0.1) ** 2
        P0 = P0.copy()
        P0[0, 0] = P0[0, 0] * sc; P0[1, 1] = P0[1, 1] * sc
        print(f"position prior sigma set to {ps:g} m", flush=True)
    # --tl-cols S: give each coefficient a prior standard deviation of
    # S / ||A_:,j||, i.e. a common prior on that column's CONTRIBUTION in nT
    # rather than on the coefficient itself. The 19 columns of the regressor
    # differ by four orders of magnitude in scale (direction cosines against
    # induced and eddy terms carrying the field magnitude), so one isotropic
    # sigma is simultaneously too tight on some blocks and too loose on others.
    if "--tl-cols" in sys.argv:
        Sc = float(sys.argv[sys.argv.index("--tl-cols") + 1])
        colnorm = np.sqrt((A ** 2).mean(axis=0))
        sd = Sc / np.maximum(colnorm, 1e-12)
        P0 = P0.copy(); P0[iTL, iTL] = np.diag(sd ** 2)
        print(f"TL prior per column: contribution sigma {Sc:g} nT, "
              f"coefficient sigma {sd.min():.3g} to {sd.max():.3g}", flush=True)
    # --sigma-s X: scale the FOGM disturbance prior and walk. The fitted S runs
    # to 70 nT against a nominal 3 nT prior in the cold-start window solves, so
    # this is the other under-scaled prior in the same measurement equation.
    if "--sigma-s" in sys.argv:
        Ss = float(sys.argv[sys.argv.index("--sigma-s") + 1])
        P0 = P0.copy(); P0[iS, iS] = P0[iS, iS] * Ss ** 2
        for s in range(M - 1):
            QdK[s][iS, iS] = QdK[s][iS, iS] * Ss ** 2
        print(f"FOGM prior and walk sigma scaled by {Ss:g}", flush=True)
    if "--tl-walk" in sys.argv:
        Wk = float(sys.argv[sys.argv.index("--tl-walk") + 1])
        for s in range(M - 1):
            QdK[s][iTL, iTL] = QdK[s][iTL, iTL] * Wk ** 2
        print(f"TL random walk sigma scaled by {Wk:g}", flush=True)

    x0_TL = np.zeros(nTL)
    if "--tl-init" in sys.argv:
        T_init = float(sys.argv[sys.argv.index("--tl-init") + 1])
        n_init = min(N, int(round(T_init / dt)))
        r0 = np.array([meas[t] - grid.value(ins_lat[t], ins_lon[t])
                       for t in range(n_init)])
        Aw = A[:n_init]
        lam = Rm / sig_TL ** 2                   # penalty = the TL prior in use
        x0_TL = np.linalg.solve(Aw.T @ Aw + lam * np.eye(nTL), Aw.T @ r0)
        print(f"TL bootstrap over first {T_init:g}s: |beta|max="
              f"{np.abs(x0_TL).max():.1f}, residual rms "
              f"{np.sqrt(((r0 - Aw @ x0_TL)**2).mean()):.1f} nT "
              f"(uncompensated {np.sqrt((r0**2).mean()):.1f} nT)", flush=True)

    dyn_nms = [gtsam.noiseModel.Gaussian.Covariance(QdK[s]) for s in range(M-1)]
    prior_nm = gtsam.noiseModel.Gaussian.Covariance(P0)
    meas_base = gtsam.noiseModel.Isotropic.Sigma(1, math.sqrt(Rm))
    # --norobust / --huber C: the Julia reference (fgo_online) applies the Huber
    # weights only from the SECOND Gauss-Newton iteration and survives Mag 4 both
    # with and without the kernel; GTSAM applies it from the first linearization,
    # where the uncompensated cabin field makes every residual an "outlier".
    if "--norobust" in sys.argv:
        meas_nm = meas_base
        print("no robust kernel (pure Gaussian measurement factors)", flush=True)
    else:
        hub_c = (float(sys.argv[sys.argv.index("--huber") + 1])
                 if "--huber" in sys.argv else 1.345)
        meas_nm = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(hub_c), meas_base)
        if hub_c != 1.345:
            print(f"Huber c={hub_c}", flush=True)

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
    # --norelin: the opposite end of the same knob, and the control the accuracy
    # claim needs. Each new state is still linearized at its initial value, which
    # is Phi times the previous causal estimate -- the point a filter would
    # linearize at -- but no state is ever relinearized afterwards. What is left
    # is linear fixed-lag smoothing over the same window: the same error model,
    # the same compensation model, the same lag, the same measurements, and none
    # of the revision the graph is credited with. Section I concedes that for a
    # linear-Gaussian model the fixed-lag MAP estimate coincides with classical
    # fixed-lag smoothing, so smoothing per se is not the contribution; this run
    # is what separates the two. The gap between this and the full run is the
    # formulation's share of the margin, and the gap between this and the causal
    # filters is the share that smoothing alone would have bought anyway.
    if "--norelin" in sys.argv:
        params.setRelinearizeThreshold(1e12)
        params.relinearizeSkip = 10**9
        print("no relinearization (linear fixed-lag smoothing control)", flush=True)
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

    # --seed-only: use the bootstrap coefficients as the initial iterate but
    # leave the prior mean at zero, so the first window's measurements are not
    # counted once in the prior and once as measurement factors.
    x_prior = np.zeros(nx); x_prior[iTL] = x0_TL
    x_prior_mean = np.zeros(nx) if "--seed-only" in sys.argv else x_prior
    prior_nm = gtsam.noiseModel.Gaussian.Covariance(P0)
    graph = gtsam.NonlinearFactorGraph(); vals = gtsam.Values(); ts = KTM()
    est = np.zeros((M, nx)); est_rt = np.zeros((M, nx))
    graph.push_back(gtsam.PriorFactorVector(X(0), x_prior_mean, prior_nm))
    vals.insert(X(0), x_prior); ts.insert((X(0), 0.0))
    est_rt[0] = x_prior
    add_meas(graph, 0)

    t0 = time.time()
    lag_states = int(round(lag / dtK))

    # --boot W: batch bootstrap. Growing the graph one state at a time walks the
    # smoother through the data-starved startup, where the window optimum is
    # kilometres from truth (diag_window1.py: 2137 m at 100 s on line 1003.02
    # Mag 4) and ISAM2 keeps that linearization for good. Solving the first W
    # states as one batch first, then handing the result to the smoother, skips
    # the ill-posed region entirely. Cost is a few seconds of startup latency,
    # after which the update stays per-epoch and real time.
    s_boot = 1
    if "--boot" in sys.argv:
        W = min(M, int(round(float(sys.argv[sys.argv.index("--boot") + 1]) / dtK)))
        bg = gtsam.NonlinearFactorGraph(); bv = gtsam.Values()
        bg.push_back(gtsam.PriorFactorVector(X(0), x_prior_mean, prior_nm))
        add_meas(bg, 0); bv.insert(X(0), x_prior)
        for s in range(1, W):
            bg.add(gtsam.CustomFactor(dyn_nms[s-1], [X(s-1), X(s)],
                                      dyn_err(PhiK[s-1])))
            add_meas(bg, s)
            bv.insert(X(s), x_prior)
        lp = gtsam.LevenbergMarquardtParams()
        lp.setLinearSolverType("MULTIFRONTAL_QR")
        lp.setMaxIterations(100)
        bres = gtsam.LevenbergMarquardtOptimizer(bg, bv, lp).optimize()
        x_boot = np.array([bres.atVector(X(s)) for s in range(W)])
        # Hand the bootstrap to the smoother one lag-length at a time: the
        # smoother cannot marginalize keys introduced in the same update, and a
        # first block of a full lag skips the data-starved region where the
        # window optimum is kilometres off. Later blocks only ever marginalize
        # keys from earlier updates.
        blk = min(W, lag_states)
        graph = gtsam.NonlinearFactorGraph(); vals = gtsam.Values(); ts = KTM()
        graph.push_back(gtsam.PriorFactorVector(X(0), x_prior_mean, prior_nm))
        add_meas(graph, 0)
        vals.insert(X(0), x_boot[0]); ts.insert((X(0), 0.0))
        for s in range(1, W):
            graph.add(gtsam.CustomFactor(dyn_nms[s-1], [X(s-1), X(s)],
                                         dyn_err(PhiK[s-1])))
            add_meas(graph, s)
            vals.insert(X(s), x_boot[s]); ts.insert((X(s), s * dtK))
            if (s + 1) % blk == 0 or s == W - 1:
                sm.update(graph, vals, ts)
                graph = gtsam.NonlinearFactorGraph()
                vals = gtsam.Values(); ts = KTM()
        cur = sm.calculateEstimate()
        for s in range(W):
            if cur.exists(X(s)):
                est[s] = cur.atVector(X(s))
            else:
                est[s] = x_boot[s]
        # inside the bootstrap window the "realtime" column is the batch value,
        # i.e. smoothed rather than causal. Every reported DRMS uses a warm-up of
        # at least twice the bootstrap length, so no reported number is computed
        # from a state inside the bootstrap window.
        est_rt[:W] = est[:W]
        s_boot = W
        print(f"batch bootstrap over first {W} states ({W*dtK:g}s): "
              f"{time.time()-t0:.0f}s", flush=True)

    # --dump-cov: record the 2x2 position marginal covariance of the causal
    # (newest) state and of the state about to leave the lag, for the flight
    # coverage diagnostic (Laplace covariance at the working linearization and
    # IRLS weights -- a diagnostic, not a formal NEES, on real data).
    dump_cov = "--dump-cov" in sys.argv
    cov_c = np.full((M, 2, 2), np.nan)
    cov_s = np.full((M, 2, 2), np.nan)

    for s in range(s_boot, M):
        graph.add(gtsam.CustomFactor(dyn_nms[s-1], [X(s-1), X(s)], dyn_err(PhiK[s-1])))
        add_meas(graph, s)
        # propagate the previous causal estimate as the new state's initial
        # (= linearization) point, as any filter does
        x_init = PhiK[s-1] @ est_rt[s-1] if s > 1 else x_prior
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
        if dump_cov:
            isam = sm.getISAM2()
            cov_c[s] = isam.marginalCovariance(X(s))[:2, :2]
            k_old = s - lag_states
            if k_old >= 0 and cur.exists(X(k_old)):
                cov_s[k_old] = isam.marginalCovariance(X(k_old))[:2, :2]
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
    extra = {"cov_c": cov_c, "cov_s": cov_s} if dump_cov else {}
    np.savez(f"research/gtsam_poc/est_gtsam{tag}.npz", est=est, est_rt=est_rt,
             idx=idx, ins_lat=il_, ins_lon=io_, true_lat=tl_, true_lon=tn_,
             **extra)

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

    # transient window [0, warm): DRMS and peak horizontal error, both outputs
    def tran_of(lat_, lon_):
        m = ti < warm
        dn = (lat_[m] - tl_[m]) * R_EARTH
        de = (lon_[m] - tn_[m]) * R_EARTH * np.cos(tl_[m])
        e = np.sqrt(dn**2 + de**2)
        return math.sqrt(np.mean(e**2)), float(e.max())
    gt, gp = tran_of(est_lat, est_lon)
    rt_, rp = tran_of(rt_lat, rt_lon)
    lines.append(f"transient[0,{warm:g}s)  smoothed_drms={gt:.2f} m "
                 f"peak={gp:.2f} m  realtime_drms={rt_:.2f} m peak={rp:.2f} m")
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
