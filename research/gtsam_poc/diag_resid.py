#!/usr/bin/env python3
"""Measurement-residual scale diagnostic for the Mag-4 divergence.

Answers two questions the divergence hypothesis chain needs:
  1. how large is the residual GTSAM sees at its FIRST linearization point
     (x = 0, i.e. INS position, zero Tolles-Lawson coefficients)?  The Huber
     kernel treats anything past 1.345*sqrt(R) as an outlier, so if the
     uncompensated cabin field is orders of magnitude past that knee, every
     measurement factor is downweighted before the calibration ever converges.
  2. what residual is achievable at the TRUE position with a least-squares TL
     fit?  That is the irreducible model error, to be compared with the assumed
     measurement standard deviation sqrt(R).

Usage: diag_resid.py <line.h5>
"""
import sys
import numpy as np
import h5py


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/cin64/magnav_data/line_1007_06_full.h5"
    d = {}
    with h5py.File(path, "r") as fh:
        for k in fh.keys():
            d[k] = fh[k][()]

    glat = d["glat"].ravel()
    glon = d["glon"].ravel()
    gh = d["gh_rowmajor"]
    A = np.asarray(d["A"], float)
    Rm = float(d["R"])
    il = d["ins_lat"].ravel()
    io = d["ins_lon"].ravel()
    tl = d["true_lat"].ravel()
    tn = d["true_lon"].ravel()

    def bilin(lat, lon):
        dla = glat[1] - glat[0]
        dlo = glon[1] - glon[0]
        i = np.clip((lat - glat[0]) / dla, 0, glat.size - 1.001)
        j = np.clip((lon - glon[0]) / dlo, 0, glon.size - 1.001)
        i0 = i.astype(int)
        j0 = j.astype(int)
        fi = i - i0
        fj = j - j0
        return (gh[i0, j0] * (1 - fi) * (1 - fj) + gh[i0 + 1, j0] * fi * (1 - fj)
                + gh[i0, j0 + 1] * (1 - fi) * fj + gh[i0 + 1, j0 + 1] * fi * fj)

    knee = 1.345 * np.sqrt(Rm)
    print(f"assumed sqrt(R) = {np.sqrt(Rm):.1f} nT   Huber knee = {knee:.1f} nT   "
          f"nTL = {A.shape[1]}")
    for mg in ("4", "5"):
        z = np.asarray(d[f"meas_mag{mg}"], float).ravel()
        for name, (la, lo) in (("INS pos", (il, io)), ("TRUE pos", (tl, tn))):
            r0 = z - bilin(la, lo)
            beta, *_ = np.linalg.lstsq(A, r0, rcond=None)
            r1 = r0 - A @ beta
            print(f"mag{mg} {name:8s}: raw resid mean {r0.mean():10.1f} "
                  f"rms {np.sqrt((r0**2).mean()):10.1f} | after LS-TL rms "
                  f"{np.sqrt((r1**2).mean()):7.1f}  p50 {np.percentile(np.abs(r1), 50):6.1f}"
                  f"  p95 {np.percentile(np.abs(r1), 95):7.1f}")
        n = 3000                      # first 300 s = one cold-start window
        r0 = z[:n] - bilin(il[:n], io[:n])
        beta, *_ = np.linalg.lstsq(A[:n], r0, rcond=None)
        r1 = r0 - A[:n] @ beta
        print(f"   mag{mg} first 300 s: raw rms {np.sqrt((r0**2).mean()):10.1f}  "
              f"after LS-TL rms {np.sqrt((r1**2).mean()):7.1f}")


if __name__ == "__main__":
    main()
