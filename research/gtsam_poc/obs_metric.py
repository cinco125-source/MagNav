#!/usr/bin/env python3
"""Whitened projected-gradient observability metric mu(W) on the flight lines.

For a window of W seconds at the 1 Hz kept cadence, stack the map-gradient
rows G (nT/m, evaluated at the true positions) and the TL regressor rows Psi
(shared by Mag 4 and Mag 5: the regressor is built from the fluxgates), whiten
both by the measurement sigma, and compute

    mu(W) = lambda_min( Gb' (I - Pb Pb^+) Gb ),   Gb = G/sigma, Pb = Psi/sigma,

the smallest singular value squared of the compensation-orthogonal position
gradient. mu^{-1/2} [m] is the weakest-direction position standard deviation a
window of length W can support under a static-coefficient reading of the
measurement model (paper Eq. eq:mu). Windows slide along the line with a 30 s
stride; the median and interquartile range over window positions are reported
per lag, per line.

Data: /home/cin64/magnav_data/line_<line>_full.h5 (export_full_line.py),
fields A [N x 19], true_lat/true_lon [rad], gh_rowmajor/glat/glon, R, dt.

Writes research/gtsam_poc/obs_metric_results.csv:
    line,lag_s,n_win,mu_med,sig_med,sig_q1,sig_q3
"""
import csv
import math
import os

import h5py
import numpy as np

R_EARTH = 6378137.0
SIG_R = 12.0            # measurement sigma used by the estimator [nT]
K = 10                  # kept cadence: one state per K samples (1 Hz)
LAGS = [30, 60, 120, 180, 240, 300, 450, 600]
STRIDE = 30             # window stride [s]
LINES = ["1007_06", "1007_02", "1003_02", "1003_08"]
DATA = "/home/cin64/magnav_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "obs_metric_results.csv")


class Grid:
    def __init__(self, glat, glon, gh):
        self.glat = np.asarray(glat).ravel()
        self.glon = np.asarray(glon).ravel()
        self.gh = np.asarray(gh)
        self.dlat = self.glat[1] - self.glat[0]
        self.dlon = self.glon[1] - self.glon[0]

    def value(self, lat, lon):
        i = np.clip((lat - self.glat[0]) / self.dlat, 0, self.glat.size - 1.001)
        j = np.clip((lon - self.glon[0]) / self.dlon, 0, self.glon.size - 1.001)
        i0 = np.asarray(i, dtype=int); j0 = np.asarray(j, dtype=int)
        fi = i - i0; fj = j - j0
        g = self.gh
        return (g[i0, j0] * (1 - fi) * (1 - fj) + g[i0 + 1, j0] * fi * (1 - fj)
                + g[i0, j0 + 1] * (1 - fi) * fj
                + g[i0 + 1, j0 + 1] * fi * fj)

    def grad(self, lat, lon):
        h = 1e-6
        return ((self.value(lat + h, lon) - self.value(lat - h, lon)) / (2 * h),
                (self.value(lat, lon + h) - self.value(lat, lon - h)) / (2 * h))


def mu_of_window(Gb, Pb):
    """lambda_min of Gb' (I - Pb Pb^+) Gb via orthonormal basis of range(Pb)."""
    U, s, _ = np.linalg.svd(Pb, full_matrices=False)
    tol = max(Pb.shape) * np.finfo(float).eps * s[0]
    Ur = U[:, s > tol]
    Gp = Gb - Ur @ (Ur.T @ Gb)
    sv = np.linalg.svd(Gp, compute_uv=False)
    return sv[-1] ** 2


def main():
    rows = []
    for line in LINES:
        path = os.path.join(DATA, f"line_{line}_full.h5")
        with h5py.File(path, "r") as f:
            A = np.asarray(f["A"], dtype=float)
            lat = np.asarray(f["true_lat"]).ravel()
            lon = np.asarray(f["true_lon"]).ravel()
            grid = Grid(f["glat"], f["glon"], f["gh_rowmajor"])
        if A.shape[0] != lat.size:
            A = A.T
        idx = np.arange(0, lat.size, K)          # kept samples, 1 Hz
        latk, lonk, Ak = lat[idx], lon[idx], A[idx]
        gl, go = grid.grad(latk, lonk)           # nT/rad
        gn = gl / R_EARTH                        # nT/m north
        ge = go / (R_EARTH * np.cos(latk))       # nT/m east
        G = np.column_stack([gn, ge]) / SIG_R
        P = Ak / SIG_R
        M = latk.size
        for L in LAGS:
            mus = []
            for s0 in range(0, M - L + 1, STRIDE):
                mus.append(mu_of_window(G[s0:s0 + L], P[s0:s0 + L]))
            mus = np.asarray(mus)
            sig = 1.0 / np.sqrt(np.maximum(mus, 1e-30))
            rows.append((line, L, len(mus), float(np.median(mus)),
                         float(np.median(sig)),
                         float(np.percentile(sig, 25)),
                         float(np.percentile(sig, 75))))
            print(f"{line} L={L:3d}s: n={len(mus):3d} "
                  f"median mu^-1/2 = {np.median(sig):8.2f} m "
                  f"(IQR {np.percentile(sig,25):.2f}-{np.percentile(sig,75):.2f})",
                  flush=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["line", "lag_s", "n_win", "mu_med",
                    "sig_med", "sig_q1", "sig_q3"])
        w.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
