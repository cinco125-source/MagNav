#!/usr/bin/env python3
"""Monte Carlo over the initial navigation error, on the real 1007.06 data.

WHY THIS AND NOT A CLASSICAL MC. Line 1007.06 is one physical realization: the
recorded INS drift, the map error and the cabin interference cannot be re-drawn
without fabricating them, which is why the manuscript keeps its MC in
simulation. But there is one quantity the estimator genuinely does not know and
that a real deployment genuinely re-draws every flight -- the navigation error
it starts from. That can be varied on recorded data without inventing a single
measurement.

The construction is exact rather than approximate. The error state obeys
x_{s+1} = Phi_s x_s + w_s, so adding a homogeneous solution to it is still a
valid error trajectory: draw d ~ N(0, P0) restricted to the seventeen inertial
states, propagate it with the SAME Phi the data was exported with, and subtract
its position components from the INS track. The recorded measurement, the map,
the Tolles-Lawson regressor and the recorded process noise are all untouched --
grid.value(ins' + x') evaluates at exactly the same latitude as before, because
x' = x + d_propagated by construction. Only the error the estimator has to
remove is different, and it is different in the way the filter's own prior says
it should be.

The draw is dominated by the velocity block (sigma = 1 m/s per axis against a
0.1 m position prior), so over ten minutes it puts the initial error anywhere
from tens to hundreds of metres. That is the point: it sweeps the transient
severity, which is the regime the whole warm-up argument has been circling. The
comparison is scored over the WHOLE segment, transient included, because that
is what the last week of this argument concluded is the honest convention.

WHAT IS PAIRED WITH WHAT. Every seed runs three estimators on identical data:
  EKF        the same 37-state model, same decimation, same relinearization
             point, causal -- a plain Kalman filter, which for this error model
             is exactly what src/ekf_online.jl computes
  FGO caus.  the incremental smoother's newest state, i.e. genuinely causal
  FGO 300s   the same graph read out 300 s behind
Because the seeds are common random numbers across estimators, the per-seed
RATIO is the statistic with the variance removed, and a sign test on thirty
paired ratios is a far sharper instrument than thirty unpaired distributions.

Usage:
  mc_1007.py <line.h5> [--seeds 30] [--lag 300] [--K 10] [--mag 5]
             [--tl-sigma 100] [--warm 0] [--out mc_1007.csv]
"""
import math
import os
import sys
import time

import numpy as np
import gtsam
import gtsam_unstable
from gtsam import symbol_shorthand

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gtsam_decimated import Grid, load          # noqa: E402

X = symbol_shorthand.X
R_EARTH = 6378137.0
HERE = os.path.dirname(os.path.abspath(__file__))


def arg(flag, default, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def drms(lat, lon, tlat, tlon, t, warm):
    m = t >= warm
    dn = (lat[m] - tlat[m]) * R_EARTH
    de = (lon[m] - tlon[m]) * R_EARTH * np.cos(tlat[m])
    return math.sqrt(np.mean(dn ** 2 + de ** 2))


def kalman(PhiK, QdK, P0, Rm, A, meas, ins_lat, ins_lon, grid, nx, iTL, iS,
           huber=0.0, iters=1):
    """Extended Kalman filter on the decimated chain.

    Same measurement Jacobian as the graph builds -- [dh/dlat, dh/dlon, 0..., A_s, 1]
    -- and the same relinearization point (the current estimate), so any
    difference against the graph's causal column is the graph's doing and not a
    modelling difference. Joseph form for the covariance because the TL block
    stays near-singular for the first minute.

    iters > 1 makes this an ITERATED EKF: the measurement update is re-solved
    with the Jacobian and predicted measurement re-evaluated at the updated
    estimate, which is the filter's own way of fixing a bad linearization. It
    matters because relinearization is the only channel through which a window
    can improve the CAUSAL state at all -- the marginal a filter carries is
    already a sufficient statistic for the past, so the window adds no
    information about the newest state, only a chance to revise the point the
    past was linearized at. A reviewer will ask why the baseline is not an IEKF,
    and if the cold-start margin survives one, that is the strongest sentence
    the paper has; if it does not, the last claim goes with it.

    huber = c applies the SAME M-estimator the graph is given, which
    src/ekf_online.jl does not have: it has no robust kernel, no innovation
    gating and no outlier rejection anywhere. That asymmetry is not controlled
    by the norelin run -- that control keeps the kernel on both sides -- so
    without this the graph carries a robust kernel into a comparison against a
    filter that has none, and part of any margin is the kernel rather than the
    formulation. GTSAM whitens the robust residual by the measurement sigma
    alone, so this does too: w = min(1, c/|resid|/sigma), applied as R/w, which
    is one IRLS step of exactly the weight fgo.jl's robust_weight computes.
    """
    M = A.shape[0]
    x = np.zeros(nx)
    P = P0.copy()
    out = np.zeros((M, nx))
    I = np.eye(nx)
    for s in range(M):
        if s > 0:
            x = PhiK[s - 1] @ x
            P = PhiK[s - 1] @ P @ PhiK[s - 1].T + QdK[s - 1]
            P = (P + P.T) / 2
        lat, lon = ins_lat[s] + x[0], ins_lon[s] + x[1]
        glat_, glon_ = grid.grad(lat, lon)
        H = np.zeros((1, nx))
        H[0, 0] = glat_
        H[0, 1] = glon_
        H[0, iTL] = A[s]
        H[0, iS] = 1.0
        h = grid.value(lat, lon) + A[s] @ x[iTL] + x[iS]
        xi = x.copy()
        for _ in range(max(1, iters) - 1):
            lat_i, lon_i = ins_lat[s] + xi[0], ins_lon[s] + xi[1]
            gl_, go_ = grid.grad(lat_i, lon_i)
            Hi = np.zeros((1, nx))
            Hi[0, 0] = gl_; Hi[0, 1] = go_
            Hi[0, iTL] = A[s]; Hi[0, iS] = 1.0
            hi = grid.value(lat_i, lon_i) + A[s] @ xi[iTL] + xi[iS]
            Si = float((Hi @ P @ Hi.T)[0, 0]) + Rm
            Ki = (P @ Hi.T / Si).ravel()
            # IEKF: innovation taken about the relinearization point, with the
            # prediction-to-iterate offset carried, per Bell & Cathey (1993)
            xn = x + Ki * (meas[s] - hi - float(Hi @ (x - xi)))
            if np.max(np.abs(xn - xi)) < 1e-12:
                xi = xn
                break
            xi = xn
        if iters > 1:
            lat, lon = ins_lat[s] + xi[0], ins_lon[s] + xi[1]
            glat_, glon_ = grid.grad(lat, lon)
            H[0, 0] = glat_; H[0, 1] = glon_
            h = grid.value(lat, lon) + A[s] @ xi[iTL] + xi[iS] + float(H @ (x - xi))
        resid = meas[s] - h
        Rw = Rm
        if huber > 0.0:
            e = abs(resid) / math.sqrt(Rm)
            if e > huber:
                Rw = Rm / max(huber / e, 1e-6)
        S = float((H @ P @ H.T)[0, 0]) + Rw
        Kg = (P @ H.T / S).ravel()
        x = x + Kg * resid
        KH = I - np.outer(Kg, H.ravel())
        P = KH @ P @ KH.T + np.outer(Kg, Kg) * Rw
        P = (P + P.T) / 2
        out[s] = x
    return out


def fgo(PhiK, QdK, P0, Rm, A, meas, ins_lat, ins_lon, grid, nx, iTL, iS,
        lag, dtK, norelin=False, huber=1.345):
    """The estimator under test: incremental fixed-lag smoother, unchanged.

    norelin=True freezes every linearization at the point a filter would pick
    (Phi times the previous causal estimate) and never revises it, which leaves
    linear fixed-lag smoothing over the same window with the same model, lag and
    measurements. The gap against the default run is the formulation's own share
    of whatever margin exists; what is left is what smoothing alone would have
    bought. Section I already concedes that smoothing per se is not a
    contribution, so this control is what separates the two."""
    M = A.shape[0]
    dyn_nms = [gtsam.noiseModel.Gaussian.Covariance(QdK[s]) for s in range(M - 1)]
    prior_nm = gtsam.noiseModel.Gaussian.Covariance(P0)
    meas_base = gtsam.noiseModel.Isotropic.Sigma(1, math.sqrt(Rm))
    meas_nm = meas_base if huber <= 0.0 else gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(huber), meas_base)

    def dyn_err(Phi_t):
        def f(this, values, J):
            xa = values.atVector(this.keys()[0])
            xb = values.atVector(this.keys()[1])
            if J is not None:
                J[0] = -Phi_t
                J[1] = np.eye(nx)
            return xb - Phi_t @ xa
        return f

    def meas_err(s):
        lat0, lon0, At, zt = ins_lat[s], ins_lon[s], A[s], meas[s]

        def f(this, values, J):
            x = values.atVector(this.keys()[0])
            lat, lon = lat0 + x[0], lon0 + x[1]
            h = grid.value(lat, lon) + At @ x[iTL] + x[iS]
            if J is not None:
                jac = np.zeros((1, nx))
                glat_, glon_ = grid.grad(lat, lon)
                jac[0, 0] = glat_
                jac[0, 1] = glon_
                jac[0, iTL] = At
                jac[0, iS] = 1.0
                J[0] = jac
            return np.array([h - zt])
        return f

    params = gtsam.ISAM2Params()
    params.setFactorization("QR")
    if norelin:
        params.setRelinearizeThreshold(1e12)
        params.relinearizeSkip = 10 ** 9
    sm = gtsam_unstable.IncrementalFixedLagSmoother(lag, params)
    KTM = gtsam_unstable.FixedLagSmootherKeyTimestampMap
    graph = gtsam.NonlinearFactorGraph()
    vals = gtsam.Values()
    ts = KTM()
    est = np.zeros((M, nx))
    est_rt = np.zeros((M, nx))
    graph.push_back(gtsam.PriorFactorVector(X(0), np.zeros(nx), prior_nm))
    graph.add(gtsam.CustomFactor(meas_nm, [X(0)], meas_err(0)))
    vals.insert(X(0), np.zeros(nx))
    ts.insert((X(0), 0.0))
    lag_states = int(round(lag / dtK))
    for s in range(1, M):
        graph.add(gtsam.CustomFactor(dyn_nms[s - 1], [X(s - 1), X(s)],
                                     dyn_err(PhiK[s - 1])))
        graph.add(gtsam.CustomFactor(meas_nm, [X(s)], meas_err(s)))
        vals.insert(X(s), PhiK[s - 1] @ est_rt[s - 1])
        ts.insert((X(s), s * dtK))
        sm.update(graph, vals, ts)
        graph = gtsam.NonlinearFactorGraph()
        vals = gtsam.Values()
        ts = KTM()
        cur = sm.calculateEstimate()
        est_rt[s] = cur.atVector(X(s))
        for k in range(max(0, s - lag_states), s + 1):
            if cur.exists(X(k)):
                est[k] = cur.atVector(X(k))
    return est, est_rt


def main():
    path = sys.argv[1]
    seeds = int(arg("--seeds", 30, int))
    lag = float(arg("--lag", 300.0, float))
    K = int(arg("--K", 10, int))
    mag = arg("--mag", "5")
    sig_TL = float(arg("--tl-sigma", 100.0, float))
    warm = float(arg("--warm", 0.0, float))
    # --scale c: multiply the drawn initial error by c. c = 1 draws from the
    # prior the filter is handed, which is dominated by its 1 m/s per-axis
    # velocity block and over ten minutes leaves an INS 600 m out -- an order
    # above this line's recorded 52 m. Sweeping c is therefore not a knob for
    # taste but the question itself: at which initial-error magnitude, if any,
    # does the graph's causal output separate from a filter's? c = 0 is the
    # recorded line untouched and is deterministic, so one seed suffices.
    scale = float(arg("--scale", 1.0, float))
    norelin = "--norelin" in sys.argv
    # --norobust drops the Huber kernel from the graph. Paired against the
    # default run it says how much of the graph's margin is the kernel that
    # src/ekf_online.jl never had, and the EKF_huber column below is the same
    # question asked from the other side.
    hub = 0.0 if "--norobust" in sys.argv else float(arg("--huber", 1.345, float))
    n_iekf = int(arg("--iekf", 5, int))
    # --pos-sigma X: initial horizontal position uncertainty in metres, applied
    # to BOTH the prior the estimators are given and the error actually drawn.
    # The export ships 0.1 m, which puts every run in the TRACKING regime: one
    # mode of the map likelihood inside the prior, so the posterior is unimodal,
    # an EKF is optimal, a particle filter has nothing to represent, and
    # relinearization has nothing to fix. That is why nothing separates. The map
    # here is 40 x 24 km with a 202 nT rms anomaly on a 200 m grid, so at
    # kilometre-scale uncertainty the likelihood is genuinely multimodal and the
    # estimator question becomes real -- which is also the operationally honest
    # cold start, INS having drifted before the map is switched on.
    pos_sigma = float(arg("--pos-sigma", 0.1, float))
    out_csv = os.path.join(HERE, arg("--out", "mc_1007_results.csv"))

    d = load(path)
    N = int(d["N"]); nx = int(d["nx"]); nTL = int(d["nTL"])
    dt = float(d["dt"]); Rm = float(d["R"])
    Phi = np.asarray(d["Phi"], dtype=float)
    A_full = np.asarray(d["A"], dtype=float)
    if A_full.shape[0] != N:                        # julia layout is [nTL, N]
        A_full = A_full.T
    key = f"meas_mag{mag}" if f"meas_mag{mag}" in d else "meas"
    meas_full = np.asarray(d[key], dtype=float).ravel()
    ins_lat_f = np.asarray(d["ins_lat"]).ravel()
    ins_lon_f = np.asarray(d["ins_lon"]).ravel()
    true_lat_f = np.asarray(d["true_lat"]).ravel()
    true_lon_f = np.asarray(d["true_lon"]).ravel()
    P0 = np.asarray(d["P0"], dtype=float)
    Qd = np.asarray(d["Qd"], dtype=float) + 1e-20 * np.eye(nx)
    gh = d["gh_rowmajor"] if "gh_rowmajor" in d else np.asarray(d["gh"])
    grid = Grid(d["glat"], d["glon"], gh)

    idx = np.arange(0, N, K)
    M = idx.size
    dtK = dt * K
    PhiK = np.zeros((M - 1, nx, nx))
    QdK = np.zeros((M - 1, nx, nx))
    for s in range(M - 1):
        P_ = np.eye(nx); Q_ = np.zeros((nx, nx))
        for t in range(idx[s], min(idx[s + 1], N - 1)):
            P_ = Phi[t] @ P_
            Q_ = Phi[t] @ Q_ @ Phi[t].T + Qd
        PhiK[s] = P_
        QdK[s] = (Q_ + Q_.T) / 2

    iTL = slice(17, 17 + nTL)
    iS = nx - 1
    P0 = P0.copy()
    P0[iTL, iTL] = P0[iTL, iTL] * sig_TL ** 2

    A = A_full[idx]
    meas = meas_full[idx]
    ins_lat0 = ins_lat_f[idx]; ins_lon0 = ins_lon_f[idx]
    tlat = true_lat_f[idx]; tlon = true_lon_f[idx]
    ti = idx * dt

    # the draw covariance: the inertial block of the prior the filter is given.
    # The compensation and disturbance states are NOT drawn -- they are
    # properties of the installation and the map, and perturbing them would
    # require perturbing the recorded measurement, which is the line this
    # construction refuses to cross.
    Pn = P0[:17, :17]
    Ln = np.linalg.cholesky(Pn + 1e-24 * np.eye(17))
    # the position draw is kept separate from --scale so the two knobs isolate:
    # --scale moves the inertial states, --pos-sigma moves where you think you are
    sig_pos_rad = pos_sigma / R_EARTH
    if pos_sigma != 0.1:
        P0 = P0.copy()
        P0[0, 0] = sig_pos_rad ** 2
        P0[1, 1] = (pos_sigma / (R_EARTH * math.cos(float(tlat[0])))) ** 2

    if scale == 0.0:
        seeds = 1
    print(f"MC over the initial navigation error: {seeds} seeds, scale={scale:g}, "
          f"lag={lag:g}s K={K} mag={mag} sigma_beta={sig_TL:g} "
          f"pos_sigma={pos_sigma:g}m "
          f"scored from t={warm:g}s over {ti[-1]:.0f}s"
          + ("  [NORELIN]" if norelin else "")
          + ("  [NOROBUST]" if hub <= 0 else f"  huber={hub:g}"),
          flush=True)
    print(f"{'seed':>4s}{'|d0| pos':>10s}{'INS':>9s}{'EKF':>9s}{'EKFhub':>9s}{'IEKF':>9s}"
          f"{'FGOcaus':>9s}{'FGO300':>9s}{'c/EKF':>8s}{'s/EKF':>8s}", flush=True)

    rows = []
    t_start = time.time()
    for sd in range(seeds):
        rng = np.random.default_rng(1000 + sd)
        dn = scale * (Ln @ rng.standard_normal(17))
        xd = np.zeros((M, nx))
        xd[0, :17] = dn
        if pos_sigma != 0.1:
            xd[0, 0] += sig_pos_rad * rng.standard_normal()
            xd[0, 1] += (pos_sigma / (R_EARTH * math.cos(float(tlat[0])))) \
                * rng.standard_normal()
        for s in range(1, M):
            xd[s] = PhiK[s - 1] @ xd[s - 1]
        # subtract the perturbation's position components from the INS track, so
        # the true error becomes x + xd and every measurement stays valid
        ins_lat = ins_lat0 - xd[:, 0]
        ins_lon = ins_lon0 - xd[:, 1]

        ins_d = drms(ins_lat, ins_lon, tlat, tlon, ti, warm)
        xk = kalman(PhiK, QdK, P0, Rm, A, meas, ins_lat, ins_lon, grid,
                    nx, iTL, iS)
        ekf_d = drms(ins_lat + xk[:, 0], ins_lon + xk[:, 1], tlat, tlon, ti, warm)
        xh = kalman(PhiK, QdK, P0, Rm, A, meas, ins_lat, ins_lon, grid,
                    nx, iTL, iS, huber=1.345)
        ekh_d = drms(ins_lat + xh[:, 0], ins_lon + xh[:, 1], tlat, tlon, ti, warm)
        xi_ = kalman(PhiK, QdK, P0, Rm, A, meas, ins_lat, ins_lon, grid,
                     nx, iTL, iS, iters=n_iekf)
        iek_d = drms(ins_lat + xi_[:, 0], ins_lon + xi_[:, 1], tlat, tlon, ti, warm)
        est, est_rt = fgo(PhiK, QdK, P0, Rm, A, meas, ins_lat, ins_lon, grid,
                          nx, iTL, iS, lag, dtK, norelin, hub)
        c_d = drms(ins_lat + est_rt[:, 0], ins_lon + est_rt[:, 1], tlat, tlon, ti, warm)
        s_d = drms(ins_lat + est[:, 0], ins_lon + est[:, 1], tlat, tlon, ti, warm)
        pos0 = math.hypot(xd[0, 0] * R_EARTH, xd[0, 1] * R_EARTH * math.cos(tlat[0]))
        rows.append((sd, pos0, ins_d, ekf_d, ekh_d, iek_d, c_d, s_d))
        print(f"{sd:4d}{pos0:10.2f}{ins_d:9.1f}{ekf_d:9.1f}{ekh_d:9.1f}{iek_d:9.1f}"
              f"{c_d:9.1f}{s_d:9.1f}{c_d/ekf_d:8.3f}{s_d/ekf_d:8.3f}", flush=True)
        if scale == 0.0:
            # the unperturbed run is the control: at warm = 300 s it must
            # reproduce the committed segment result (15.49 causal / 7.88
            # smoothed) or the harness differs from the runner it claims to be
            for wv in (60.0, 300.0):
                print(f"      control warm={wv:g}s  ins {drms(ins_lat, ins_lon, tlat, tlon, ti, wv):8.2f}"
                      f"  ekf {drms(ins_lat+xk[:,0], ins_lon+xk[:,1], tlat, tlon, ti, wv):8.2f}"
                      f"  caus {drms(ins_lat+est_rt[:,0], ins_lon+est_rt[:,1], tlat, tlon, ti, wv):8.2f}"
                      f"  smoo {drms(ins_lat+est[:,0], ins_lon+est[:,1], tlat, tlon, ti, wv):8.2f}",
                      flush=True)

    with open(out_csv, "w") as f:
        f.write("seed,pos0_m,INS,EKF,EKF_huber,IEKF,FGO_causal,FGO_smoothed\n")
        for r in rows:
            f.write("%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" % r)

    a = np.array([[r[3], r[6], r[7], r[4], r[5]] for r in rows])
    rc = a[:, 1] / a[:, 0]
    rs = a[:, 2] / a[:, 0]
    print()
    print(f"{seeds} seeds in {time.time()-t_start:.0f}s -> {out_csv}")
    print(f"  median DRMS   EKF {np.median(a[:,0]):.1f}   "
          f"EKF+huber {np.median(a[:,3]):.1f}   "
          f"FGO causal {np.median(a[:,1]):.1f}   FGO 300 s {np.median(a[:,2]):.1f} m")
    rh = a[:, 3] / a[:, 0]
    ri = a[:, 4] / a[:, 0]
    rci = a[:, 1] / a[:, 4]
    print(f"  huber on the FILTER alone: geo-mean {math.exp(np.log(rh).mean()):.3f}  "
          f"wins {int((rh < 1).sum())}/{rh.size}")
    print(f"  IEKF ({n_iekf} iters) / EKF:  geo-mean {math.exp(np.log(ri).mean()):.3f}  "
          f"wins {int((ri < 1).sum())}/{ri.size}   median {np.median(a[:,4]):.1f} m")
    print(f"  ours causal / IEKF:       geo-mean {math.exp(np.log(rci).mean()):.3f}  "
          f"wins {int((rci < 1).sum())}/{rci.size}   <- the one that matters")
    for lbl, r in (("causal", rc), ("smoothed", rs)):
        wins = int((r < 1).sum())
        gm = math.exp(np.log(r).mean())
        # exact two-sided sign test against p = 1/2
        n = r.size
        k = min(wins, n - wins)
        p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
        lo, hi = np.percentile(r, [5, 95])
        print(f"  ours {lbl:8s}/EKF: geo-mean {gm:.3f}  wins {wins}/{n}  "
              f"p={min(p,1.0):.4f}  [{lo:.2f}, {hi:.2f}] over seeds")


if __name__ == "__main__":
    main()
