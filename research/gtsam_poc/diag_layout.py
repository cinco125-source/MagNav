#!/usr/bin/env python3
"""Diagnose HDF5 (Julia column-major) axis-order bugs in line_1007_06.h5.

1) Phi slice orientation: the Pinson lat-row/vn-column entry is ~dt/R_mer
   (~1.6e-8 for dt=0.1). Check whether it lands at [0,3] (correct) or [3,0]
   (transposed slice).
2) Map grid orientation: correlate meas with grid(true_pos) under gh vs gh.T —
   the correct orientation must track the measurement far better.
"""
import numpy as np, h5py

f = h5py.File("research/gtsam_poc/line_1007_06.h5", "r")
N = int(f["N"][()]); nx = int(f["nx"][()]); dt = float(f["dt"][()])
Phi = np.asarray(f["Phi"], dtype=float)
if Phi.shape[0] == nx:
    Phi = np.moveaxis(Phi, 2, 0)
print("Phi shape:", Phi.shape)
P = Phi[0]
print("Phi[0][0,3] =", P[0, 3], "  Phi[0][3,0] =", P[3, 0], "  dt/R =", dt/6378137.0)
print("Phi[0][1,4] =", P[1, 4], "  Phi[0][4,1] =", P[4, 1])
off = P - np.eye(nx)
print("upper-tri |offdiag| sum:", np.abs(np.triu(off, 1)).sum(),
      "  lower-tri:", np.abs(np.tril(off, -1)).sum())

glat = np.asarray(f["glat"]).ravel(); glon = np.asarray(f["glon"]).ravel()
gh = np.asarray(f["gh"], dtype=float)
tlat = np.asarray(f["true_lat"]).ravel(); tlon = np.asarray(f["true_lon"]).ravel()
meas = np.asarray(f["meas"]).ravel()

def bilin(g, lat, lon):
    i = np.clip((lat - glat[0]) / (glat[1]-glat[0]), 0, glat.size - 1.001)
    j = np.clip((lon - glon[0]) / (glon[1]-glon[0]), 0, glon.size - 1.001)
    i0 = i.astype(int); j0 = j.astype(int); fi = i - i0; fj = j - j0
    return (g[i0, j0]*(1-fi)*(1-fj) + g[i0+1, j0]*fi*(1-fj)
            + g[i0, j0+1]*(1-fi)*fj + g[i0+1, j0+1]*fi*fj)

for name, g in (("gh as-is", gh), ("gh.T", gh.T)):
    h = bilin(g, tlat, tlon)
    r = meas - h
    cc = np.corrcoef(np.diff(h), np.diff(meas))[0, 1]
    print(f"{name}: corr(diff(h),diff(meas)) = {cc:+.4f}   "
          f"resid std = {np.std(r):8.2f} nT   h std = {np.std(h):8.2f} nT")
print("meas std:", np.std(meas))
