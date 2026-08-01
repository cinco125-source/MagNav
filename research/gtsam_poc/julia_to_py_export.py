#!/usr/bin/env python3
"""Convert the committed Julia segment export to the Python export layout.

research/gtsam_poc/line_1007_06.h5 is the original export_line.jl artifact, the
only real SGL segment carried in the repository. run_gtsam.py reads it directly
because it carries the axis fixes inline, but run_gtsam_decimated.py and
run_gtsam_split.py assume export_full_line.py's layout and would silently read
it wrong. Rather than teach the two production scripts a second format, convert
once.

Three fixes, all of them the ones run_gtsam.py documents:
  Phi  h5py hands back Julia's column-major [nx,nx,N-1] as [N-1,nx,nx] with
       every slice transposed (dt/R shows up at [3,0] instead of [0,3])
  A    stored [nTL,N], wanted [N,nTL]
  gh   stored [lat,lon] column-major, so h5py returns the transpose; the check
       passes silently on this square 200x200 grid, which is how it went
       unnoticed once before (residual std 242 nT wrong against 31 nT right)

The converted file gains gh_rowmajor, which is what the two scripts use to
detect a Python export, and meas_mag5 aliasing meas, since the segment carries
the single cabin Mag 5 channel.

Usage: python julia_to_py_export.py [in.h5] [out.h5]
"""
import sys
import numpy as np
import h5py

src = sys.argv[1] if len(sys.argv) > 1 else "research/gtsam_poc/line_1007_06.h5"
dst = sys.argv[2] if len(sys.argv) > 2 else "/tmp/line_1007_06_py.h5"

with h5py.File(src, "r") as f:
    d = {k: f[k][()] for k in f.keys()}

N, nx, nTL = int(d["N"]), int(d["nx"]), int(d["nTL"])
assert "gh_rowmajor" not in d, "already a Python export"

Phi = np.asarray(d["Phi"], dtype=float)
if Phi.shape[0] == nx:                       # [nx,nx,N-1]
    Phi = np.moveaxis(Phi, 2, 0)
else:
    Phi = Phi.transpose(0, 2, 1)             # undo the per-slice transpose
assert Phi.shape == (N-1, nx, nx), Phi.shape

A = np.asarray(d["A"], dtype=float)
if A.shape[0] != N:
    A = A.T
assert A.shape == (N, nTL), A.shape

gh = np.asarray(d["gh"], dtype=float).T      # -> (lat, lon) row-major
glat = np.asarray(d["glat"]).ravel(); glon = np.asarray(d["glon"]).ravel()
assert gh.shape == (glat.size, glon.size)

d["Phi"] = Phi; d["A"] = A
d["gh_rowmajor"] = gh
d.pop("gh", None)
meas = np.asarray(d["meas"], dtype=float).ravel()
d["meas"] = meas
d.setdefault("meas_mag5", meas)

with h5py.File(dst, "w") as f:
    for k, v in d.items():
        f[k] = v

# the transpose is the fix that fails silently, so check it against the data:
# the post-fit residual of the map against the measurement is ~31 nT when the
# grid is read correctly and ~242 nT when it is not (README, fix 2)
def bilin(F, lat, lon):
    i = np.clip((lat-glat[0])/(glat[1]-glat[0]), 0, glat.size-1.001)
    j = np.clip((lon-glon[0])/(glon[1]-glon[0]), 0, glon.size-1.001)
    i0 = i.astype(int); j0 = j.astype(int); fi = i-i0; fj = j-j0
    return (F[i0,j0]*(1-fi)*(1-fj)+F[i0+1,j0]*fi*(1-fj)
            + F[i0,j0+1]*(1-fi)*fj+F[i0+1,j0+1]*fi*fj)

tl = np.asarray(d["true_lat"]).ravel(); to = np.asarray(d["true_lon"]).ravel()
for name, F in (("transposed (used)", gh), ("as-is (wrong)", gh.T)):
    r = meas - bilin(F, tl, to)
    print(f"  map residual std, {name:18s}: {r.std():7.1f} nT")
print(f"wrote {dst}  N={N} nx={nx} nTL={nTL} dt={float(d['dt'])}")
