#!/usr/bin/env python3
"""Measure the noise model against the data, and the achieved error against the
information it implies.

Every knob tried on this estimator moves it by a few percent -- the noise model,
the relinearization schedule, the compensation parameterization. That pattern
has one explanation worth testing before another knob is tried: the estimator is
already close to the information bound, and there is nothing left for a knob to
buy. This script measures both halves of that.

WHAT IT MEASURES

1. The noise the model is told about, against the noise the data shows.
   Subtract the map at the TRUE position and fit a STATIC 19-dof Tolles-Lawson
   basis by least squares. No random walk, no disturbance state, no position
   freedom: whatever survives is field the map-plus-TL model class cannot
   explain. Its standard deviation is the honest measurement sigma, its
   autocorrelation gives the disturbance correlation time, and the
   sample-to-sample difference gives the white floor underneath it.

2. The bound those numbers imply. mu(W) = lambda_min of the whitened
   map-gradient Gram with the Tolles-Lawson subspace projected out (the metric
   of obs_metric.py, paper Eq. mu), evaluated over the estimator's own lag.
   mu^-1/2 is the weakest-direction position standard deviation the window can
   support. Reported at the assumed sigma and at the measured one.

3. The ratio the answer turns on: achieved DRMS over the bound. Near one means
   the case is finished and no modelling change will help. Well above one means
   the case is where the remaining work is.

ON LINE 1007.06 Mag 5, the 10-minute committed segment:

  meas - map(truth)                     31.4 nT
  after a static 19-dof TL fit           4.40 nT   <- the honest sigma
    autocorrelation                     +0.87 @1s, +0.54 @5s, +0.24 @30s
    FOGM tau fitted                     35 s       (the model uses 180 s)
    white floor from sample differences  0.24 nT
  bound at sigma = 12.0 (assumed)       20.24 m
  bound at sigma =  4.40 (measured)      7.42 m
  achieved smoothed                      7.88 m   -> 1.06x the bound
  achieved with the measured FOGM        7.50 m   -> 1.01x the bound

So the smoother is at the information limit on this case, which is why retuning
the disturbance to its measured (sigma, tau) is worth 4.8% and lowering R is
worth nothing: R is not the sensor, it is a lumped map-error budget, and
shrinking it to the white floor costs accuracy rather than gaining it
(0.24 nT gives 9.48 m against 7.88). The causal estimate sits at 2.1x the bound,
which is what one-sided information should give against a two-sided window.

Usage: noise_model_audit.py <line.h5> [--mag 5] [--lag 300] [--K 10]
                            [--est est_gtsam_TAG.npz] [--warm 300]
"""
import sys

import numpy as np
import h5py

R_EARTH = 6378137.0


def argv_get(flag, default, cast=float):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


class Grid:
    def __init__(self, glat, glon, gh):
        self.glat = np.asarray(glat).ravel()
        self.glon = np.asarray(glon).ravel()
        self.gh = np.asarray(gh, dtype=float)
        assert self.gh.shape == (self.glat.size, self.glon.size)

    def value(self, lat, lon):
        g, glat, glon = self.gh, self.glat, self.glon
        i = np.clip((lat - glat[0]) / (glat[1] - glat[0]), 0, glat.size - 1.001)
        j = np.clip((lon - glon[0]) / (glon[1] - glon[0]), 0, glon.size - 1.001)
        i0 = np.asarray(i).astype(int); j0 = np.asarray(j).astype(int)
        fi, fj = i - i0, j - j0
        return (g[i0, j0]*(1-fi)*(1-fj) + g[i0+1, j0]*fi*(1-fj)
                + g[i0, j0+1]*(1-fi)*fj + g[i0+1, j0+1]*fi*fj)

    def grad(self, lat, lon, h=1e-6):
        return ((self.value(lat+h, lon) - self.value(lat-h, lon)) / (2*h),
                (self.value(lat, lon+h) - self.value(lat, lon-h)) / (2*h))


def mu_of_window(Gb, Pb):
    """lambda_min of Gb' (I - Pb Pb^+) Gb (obs_metric.py)."""
    U, s, _ = np.linalg.svd(Pb, full_matrices=False)
    Ur = U[:, s > max(Pb.shape) * np.finfo(float).eps * s[0]]
    return np.linalg.svd(Gb - Ur @ (Ur.T @ Gb), compute_uv=False)[-1] ** 2


def main():
    path = sys.argv[1]
    mag = argv_get("--mag", "5", str)
    lag = argv_get("--lag", 300.0)
    K = argv_get("--K", 10, int)
    warm = argv_get("--warm", 300.0)
    est_path = argv_get("--est", None, str)

    with h5py.File(path, "r") as f:
        A = np.asarray(f["A"], dtype=float)
        key = f"meas_mag{mag}" if f"meas_mag{mag}" in f else "meas"
        meas = np.asarray(f[key], dtype=float).ravel()
        tlat = np.asarray(f["true_lat"]).ravel()
        tlon = np.asarray(f["true_lon"]).ravel()
        ghk = "gh_rowmajor" if "gh_rowmajor" in f else "gh"
        gh = np.asarray(f[ghk], dtype=float)
        if ghk == "gh":
            gh = gh.T                      # Julia column-major
        grid = Grid(f["glat"], f["glon"], gh)
        dt = float(f["dt"][()]); sigR = float(np.sqrt(f["R"][()]))
    if A.shape[0] != tlat.size:
        A = A.T

    print(f"{path}  mag {mag}  N={tlat.size}  dt={dt}  lag={lag:g}s  K={K}")
    print()
    print("-- 1. the noise the data shows ------------------------------------")
    y = meas - grid.value(tlat, tlon)
    print(f"  meas - map(truth)                  {y.std():8.2f} nT")
    r = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    r = r - r.mean()
    sig_meas = r.std()
    print(f"  after a static 19-dof TL fit       {sig_meas:8.2f} nT   <- honest sigma")
    lags = np.arange(1, int(120 / dt))
    ac = np.array([np.corrcoef(r[:-L], r[L:])[0, 1] for L in lags])
    ok = ac > 0.05
    tau = float("nan")
    if ok.sum() > 3:
        tau = -1.0 / np.polyfit(lags[ok] * dt, np.log(ac[ok]), 1)[0]
    for s_ in (1, 5, 30, 60):
        L = int(s_ / dt)
        if L < r.size:
            print(f"    rho({s_:3d}s) {np.corrcoef(r[:-L], r[L:])[0,1]:+7.3f}")
    white = np.diff(r).std() / np.sqrt(2)
    print(f"  fitted FOGM tau                    {tau:8.1f} s")
    print(f"  white floor (sample differences)   {white:8.2f} nT")
    print(f"  model in use: sqrt(R) {sigR:.1f} nT, FOGM (sigma 3.0 nT, tau 180 s)")

    print()
    print("-- 2. the bound that implies --------------------------------------")
    idx = np.arange(0, tlat.size, K)
    la, lo = tlat[idx], tlon[idx]
    gl, go = grid.grad(la, lo)
    G0 = np.column_stack([gl / R_EARTH, go / (R_EARTH * np.cos(la))])
    Ak = A[idx]
    L = int(round(lag / (dt * K)))
    bounds = {}
    for s_ in sorted({sigR, round(sig_meas, 2), 1.0}):
        mus = [mu_of_window(G0[i:i+L] / s_, Ak[i:i+L] / s_)
               for i in range(0, G0.shape[0] - L + 1, max(1, L // 10))]
        if not mus:
            print(f"  sigma {s_:6.2f} nT : window longer than the line, skipped")
            continue
        sg = 1 / np.sqrt(np.maximum(np.asarray(mus), 1e-30))
        bounds[s_] = float(np.median(sg))
        tagged = "  <- assumed" if s_ == sigR else (
            "  <- measured" if abs(s_ - sig_meas) < 0.02 else "")
        print(f"  sigma {s_:6.2f} nT : mu^-1/2 = {np.median(sg):7.2f} m "
              f"(IQR {np.percentile(sg,25):.1f}-{np.percentile(sg,75):.1f}){tagged}")

    if est_path:
        print()
        print("-- 3. achieved against the bound ----------------------------------")
        d = np.load(est_path)
        ii = d["idx"] if "idx" in d.files else idx
        m = (ii * dt) >= warm
        # run_gtsam_decimated.py ships the error states and the INS; run_gtsam.py
        # and run_gtsam_split.py ship the summed positions. Accept either.
        if "est_lat" in d.files:
            pos = (("smoothed", d["est_lat"], d["est_lon"]),
                   ("causal", d["rt_lat"], d["rt_lon"]))
        else:
            il, io = d["ins_lat"], d["ins_lon"]
            pos = (("smoothed", il + d["est"][:, 0], io + d["est"][:, 1]),
                   ("causal", il + d["est_rt"][:, 0], io + d["est_rt"][:, 1]))
        out = {}
        for nm, la_, lo_ in pos:
            dn = (la_[m] - tlat[ii][m]) * R_EARTH
            de = (lo_[m] - tlon[ii][m]) * R_EARTH * np.cos(tlat[ii][m])
            out[nm] = float(np.sqrt(np.mean(dn**2 + de**2)))
        b = bounds.get(round(sig_meas, 2))
        for nm, v in out.items():
            extra = f"   {v/b:5.2f}x the measured-sigma bound" if b else ""
            print(f"  {nm:9s} DRMS (warm={warm:g}s) {v:8.2f} m{extra}")
        if b:
            print()
            print("  near 1.0 means the case is information-limited and no modelling")
            print("  change will help; well above 1.0 is where work remains.")


if __name__ == "__main__":
    main()
