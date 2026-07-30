#!/usr/bin/env python3
"""Round-4 red-team numeric checks.

(1) Column-normalized singular-value ratio of a 300 s window of Psi (the raw
    ratio is a units artifact); (2) free-inertial DRMS and peak over the first
    600 s per line; (3) tolerance sensitivity of mu at W=300 s.
"""
import numpy as np, h5py

DATA = "/home/cin64/magnav_data"
LINES = ["1007_06", "1007_02", "1003_02", "1003_08"]
R_EARTH = 6378137.0

# ---- (1) normalized-column svals, 300 s window @ 1 Hz, line 1007.06 -------
with h5py.File(f"{DATA}/line_1007_06_full.h5", "r") as f:
    A = np.asarray(f["A"], dtype=float)
    lat = np.asarray(f["true_lat"]).ravel(); lon = np.asarray(f["true_lon"]).ravel()
    il = np.asarray(f["ins_lat"]).ravel(); io = np.asarray(f["ins_lon"]).ravel()
if A.shape[0] != lat.size: A = A.T
W = A[0:3000:10]                       # 300 rows @ 1 Hz
Wn = W / np.linalg.norm(W, axis=0, keepdims=True)
s = np.linalg.svd(Wn, compute_uv=False)
print(f"(1) normalized 300s window: smax={s[0]:.3f} smin3={s[-3]:.2e} "
      f"{s[-2]:.2e} {s[-1]:.2e}  ratio smin/smax={s[-1]/s[0]:.2e}")

# ---- (2) INS transient [0,600 s) per line ---------------------------------
print("(2) free-inertial first 600 s:")
for line in LINES:
    with h5py.File(f"{DATA}/line_{line}_full.h5", "r") as f:
        tl = np.asarray(f["true_lat"]).ravel(); tn = np.asarray(f["true_lon"]).ravel()
        il = np.asarray(f["ins_lat"]).ravel(); io = np.asarray(f["ins_lon"]).ravel()
        dt = float(f["dt"][()])
    n = int(round(600.0 / dt))
    dn = (il[:n] - tl[:n]) * R_EARTH
    de = (io[:n] - tn[:n]) * R_EARTH * np.cos(tl[:n])
    e = np.hypot(dn, de)
    print(f"    {line}: DRMS={np.sqrt(np.mean(e**2)):7.2f} m  peak={e.max():7.2f} m")

# ---- (3) tolerance sensitivity of mu at W=300, line 1007.06 ---------------
class Grid:
    def __init__(self, glat, glon, gh):
        self.glat = np.asarray(glat).ravel(); self.glon = np.asarray(glon).ravel()
        self.gh = np.asarray(gh)
        self.dlat = self.glat[1]-self.glat[0]; self.dlon = self.glon[1]-self.glon[0]
    def value(self, lat, lon):
        i = np.clip((lat-self.glat[0])/self.dlat, 0, self.glat.size-1.001)
        j = np.clip((lon-self.glon[0])/self.dlon, 0, self.glon.size-1.001)
        i0 = np.asarray(i,dtype=int); j0 = np.asarray(j,dtype=int)
        fi, fj = i-i0, j-j0; g = self.gh
        return (g[i0,j0]*(1-fi)*(1-fj)+g[i0+1,j0]*fi*(1-fj)
                +g[i0,j0+1]*(1-fi)*fj+g[i0+1,j0+1]*fi*fj)
    def grad(self, lat, lon):
        h = 1e-6
        return ((self.value(lat+h,lon)-self.value(lat-h,lon))/(2*h),
                (self.value(lat,lon+h)-self.value(lat,lon-h))/(2*h))

with h5py.File(f"{DATA}/line_1007_06_full.h5", "r") as f:
    grid = Grid(f["glat"], f["glon"], f["gh_rowmajor"])
idx = np.arange(0, lat.size, 10)
latk, lonk, Ak = lat[idx], lon[idx], A[idx]
gl, go = grid.grad(latk, lonk)
G = np.column_stack([gl/R_EARTH, go/(R_EARTH*np.cos(latk))]) / 12.0
P = Ak / 12.0
print("(3) tolerance sensitivity, W=300, per-window median mu^-1/2 [m]:")
for scale in (1e-2, 1.0, 1e2):
    sig = []
    for s0 in range(0, latk.size-300+1, 30):
        Pb, Gb = P[s0:s0+300], G[s0:s0+300]
        U, sv, _ = np.linalg.svd(Pb, full_matrices=False)
        tol = max(Pb.shape)*np.finfo(float).eps*sv[0]*scale
        Ur = U[:, sv > tol]
        Gp = Gb - Ur @ (Ur.T @ Gb)
        sig.append(1.0/np.linalg.svd(Gp, compute_uv=False)[-1])
    print(f"    tol x{scale:g}: median {np.median(sig):.2f}")
