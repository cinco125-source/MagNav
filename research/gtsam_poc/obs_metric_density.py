#!/usr/bin/env python3
"""Does denser measurement sampling shorten the window that separability needs?

Recomputes mu(W) of obs_metric.py on line 1007.06 at the 1 Hz kept cadence and
at the full 10 Hz raw cadence, over the same window lengths in SECONDS. If
separability were limited by sample count, the 10 Hz curve would sit a factor
sqrt(10) ~ 3.2 below the 1 Hz curve and a short window would become usable.
That comparison assumes independent measurement noise, which the spatially
correlated map error violates; the measured ablation (all ten samples per
state) is the empirical counterpart.
"""
import h5py
import numpy as np

DATA = "/home/cin64/magnav_data/line_1007_06_full.h5"
R_EARTH = 6378137.0
SIG_R = 12.0
LAGS_S = [30, 60, 120, 300, 600]
STRIDE_S = 30


class Grid:
    def __init__(self, glat, glon, gh):
        self.glat = np.asarray(glat).ravel(); self.glon = np.asarray(glon).ravel()
        self.gh = np.asarray(gh)
        self.dlat = self.glat[1] - self.glat[0]
        self.dlon = self.glon[1] - self.glon[0]

    def value(self, lat, lon):
        i = np.clip((lat - self.glat[0]) / self.dlat, 0, self.glat.size - 1.001)
        j = np.clip((lon - self.glon[0]) / self.dlon, 0, self.glon.size - 1.001)
        i0 = np.asarray(i, dtype=int); j0 = np.asarray(j, dtype=int)
        fi, fj = i - i0, j - j0; g = self.gh
        return (g[i0, j0] * (1 - fi) * (1 - fj) + g[i0 + 1, j0] * fi * (1 - fj)
                + g[i0, j0 + 1] * (1 - fi) * fj + g[i0 + 1, j0 + 1] * fi * fj)

    def grad(self, lat, lon):
        h = 1e-6
        return ((self.value(lat + h, lon) - self.value(lat - h, lon)) / (2 * h),
                (self.value(lat, lon + h) - self.value(lat, lon - h)) / (2 * h))


def mu(Gb, Pb):
    U, s, _ = np.linalg.svd(Pb, full_matrices=False)
    Ur = U[:, s > max(Pb.shape) * np.finfo(float).eps * s[0]]
    Gp = Gb - Ur @ (Ur.T @ Gb)
    return np.linalg.svd(Gp, compute_uv=False)[-1] ** 2


with h5py.File(DATA, "r") as f:
    A = np.asarray(f["A"], dtype=float)
    lat = np.asarray(f["true_lat"]).ravel(); lon = np.asarray(f["true_lon"]).ravel()
    grid = Grid(f["glat"], f["glon"], f["gh_rowmajor"])
    dt = float(f["dt"][()])
if A.shape[0] != lat.size:
    A = A.T

print(f"raw rate {1/dt:.0f} Hz, N={lat.size}")
for K, tag in ((10, "1 Hz states"), (1, "10 Hz raw")):
    idx = np.arange(0, lat.size, K)
    latk, lonk, Ak = lat[idx], lon[idx], A[idx]
    gl, go = grid.grad(latk, lonk)
    G = np.column_stack([gl / R_EARTH, go / (R_EARTH * np.cos(latk))]) / SIG_R
    P = Ak / SIG_R
    rate = 1.0 / (dt * K)
    out = []
    for Ls in LAGS_S:
        W = int(Ls * rate); st = max(1, int(STRIDE_S * rate))
        sig = [1 / np.sqrt(max(mu(G[s0:s0 + W], P[s0:s0 + W]), 1e-30))
               for s0 in range(0, latk.size - W + 1, st)]
        out.append(f"{Ls:4d}s: {np.median(sig):8.2f} m")
    print(f"  {tag:12s} " + "  ".join(out))
