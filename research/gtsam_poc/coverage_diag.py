#!/usr/bin/env python3
"""Flight coverage diagnostic from a --dump-cov run.

Reads est_gtsam<tag>.npz (with cov_c/cov_s), computes the horizontal-position
Mahalanobis distance of the causal and smoothed errors against their Laplace
marginal covariances, and reports the fraction of epochs inside the 95%
chi-square-2 ellipse (5.991) after the 600 s warm-up. On real data this is a
diagnostic, not a formal NEES: the model is imperfect (map error, non-Gaussian
interference, IRLS weights), so read it as "how honest is the reported
covariance in flight".

Usage: coverage_diag.py <tag> [warm=600]
"""
import sys

import numpy as np

R_EARTH = 6378137.0
CHI2_95 = 5.991


def main():
    tag = sys.argv[1]
    warm = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
    d = np.load(f"research/gtsam_poc/est_gtsam{tag}.npz")
    est, est_rt = d["est"], d["est_rt"]
    cov_c, cov_s = d["cov_c"], d["cov_s"]
    tl, tn = d["true_lat"], d["true_lon"]
    il, io = d["ins_lat"], d["ins_lon"]
    ti = d["idx"] * 0.1
    m = ti >= warm

    out = {}
    for name, e_, c_ in (("causal", est_rt, cov_c), ("smoothed", est, cov_s)):
        lat = il + e_[:, 0]; lon = io + e_[:, 1]
        dn = (lat - tl) * R_EARTH
        de = (lon - tn) * R_EARTH * np.cos(tl)
        err = np.column_stack([e_[:, 0], e_[:, 1]])  # rad, matches cov units
        ok = m & np.isfinite(c_[:, 0, 0])
        d2 = np.full(err.shape[0], np.nan)
        for k in np.flatnonzero(ok):
            de_k = np.array([lat[k] - tl[k], lon[k] - tn[k]])
            d2[k] = de_k @ np.linalg.solve(c_[k], de_k)
        v = d2[ok]
        out[name] = (np.mean(v <= CHI2_95), np.mean(v), np.median(v), v.size)
        print(f"{name:9s} coverage@95% = {out[name][0]*100:5.1f}%   "
              f"mean d2 = {out[name][1]:7.2f}  median = {out[name][2]:6.2f} "
              f"(n={out[name][3]}, expect 2.0 if consistent)")


if __name__ == "__main__":
    main()
