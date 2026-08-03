#!/usr/bin/env python3
"""How much does the compensation actually vary, and is that variation earning
its nineteen states per epoch?

Contribution 1 of the manuscript is that the Tolles-Lawson coefficients are
carried as time-varying graph variables so the compensation adapts along the
flight. That is a claim about the state model rather than the optimizer, so
none of the norelin / norobust / IEKF controls touch it -- but it had never
been measured, and "adapts along the flight" is the kind of phrasing a reviewer
asks for a number behind.

MEASURED, on the 1007.06 segment (Mag 5, sigma_beta = 100, lag 300 s, smoothed
output, scored past the transient at t >= 300 s):

  col        mean   sd over t   sd/|mean|
    1      -51.69      0.9475      0.018
    2      176.0       0.3612      0.002
    3       59.46      0.8142      0.014
    5     -200.4       0.4167      0.002
   19        7.468     1.121       0.150

  compensation A.beta   rms 270.3 nT, range 130.1 nT
  time-variation buys   rms   2.86 nT, max |diff| 5.7 nT
  measurement sigma         12.0 nT
  FOGM S                rms   0.11 nT, range 0.3 nT

So beta is very nearly constant: the coefficients move by 0.2 to 1.8 percent of
their own size, and the entire difference between the time-varying compensation
and a single frozen mean is a QUARTER of the measurement noise. The 270 nT of
motion in A.beta is A moving with attitude, not beta. The FOGM disturbance
state is doing essentially nothing at all.

That agrees with the sparse-instantiation sweep, where a single beta node for
the whole segment scored 7.80 against the per-epoch reference's 7.88: fewer
nodes did not hurt because beta never needed them.

THE OBVIOUS CONCLUSION IS WRONG, AND THAT IS THE RESULT. If beta is constant,
freezing it at its smoothed mean should help -- nineteen nuisance dimensions
leave the second pass, and the projected bound says that subspace costs 1.2x
(1.69 against 1.45 m per nT). Run it (run_gtsam_decimated.py --beta-from) and
the opposite happens:

                              smoothed   causal
  baseline, beta free            7.88     15.49
  beta frozen at smoothed mean   8.92     17.06     +13% / +10%

The 2.86 nT of wiggle is not noise. It is absorbing local map error that
otherwise gets charged to position through the map gradient, so removing it
costs far more than its size suggests. The magnitudes line up: at 1.69 m per nT
a 2.86 nT misattribution is at most 4.8 m, and the observed loss is 1.0 to
1.6 m, i.e. only the component along the position-sensitive direction bites.
It also brackets correctly against the earlier control -- freezing beta at ZERO
diverges past 10 km, freezing it at the right value costs 13 percent.

So the defensible form of Contribution 1 is not "the compensation adapts along
the flight" but:

  the time-varying part of the compensation is small -- 2.86 nT rms, a quarter
  of the measurement noise -- and removing it costs 13 percent of position
  error, because it is anti-correlated with local map error rather than random.

One segment, one magnetometer, one straight line. Over a full line or across
lines beta has more reason to move (temperature, currents, permanent field), so
repeating this on 87 minutes would strengthen it considerably. --beta-from is
in place for that.

Usage: beta_variation.py <est_gtsam_TAG.npz> <line.h5> [--warm 300]
"""
import os
import sys

import h5py
import numpy as np


def arg(flag, default, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main():
    if len(sys.argv) < 3:
        print(__doc__.strip().splitlines()[-1])
        return 1
    z = np.load(sys.argv[1])
    f = h5py.File(sys.argv[2], "r")
    warm = float(arg("--warm", 300.0, float))
    nTL = int(f["nTL"][()])
    dt = float(f["dt"][()])
    A = np.asarray(f["A"])
    if A.shape[0] != int(f["N"][()]):
        A = A.T
    A = A[z["idx"]]
    est = z["est"]
    b = est[:, 17:17 + nTL]
    S = est[:, -1]
    t = z["idx"] * dt
    m = t >= warm

    print(f"{os.path.basename(sys.argv[1])}, smoothed, t >= {warm:g} s")
    print(f"{'col':>4}{'mean':>12}{'sd over t':>12}{'sd/|mean|':>11}{'drift':>12}")
    for j in list(range(min(6, nTL))) + [nTL - 1]:
        bb = b[m, j]
        print(f"{j+1:>4}{bb.mean():12.4g}{bb.std():12.4g}"
              f"{bb.std()/max(abs(bb.mean()),1e-12):11.3g}{bb[-1]-bb[0]:12.4g}")

    pred_t = np.einsum("tj,tj->t", A, b)
    pred_c = A @ b[m].mean(axis=0)
    d = pred_t[m] - pred_c[m]
    sig = float(np.sqrt(f["R"][()]))
    print()
    print(f"compensation A.beta : rms {np.sqrt((pred_t[m]**2).mean()):8.1f} nT"
          f"   range {pred_t[m].max()-pred_t[m].min():.1f} nT")
    print(f"time-variation buys : rms {np.sqrt((d**2).mean()):8.2f} nT"
          f"   max |diff| {np.abs(d).max():.1f} nT")
    print(f"measurement sigma   : {sig:12.1f} nT")
    print(f"FOGM S              : rms {np.sqrt((S[m]**2).mean()):8.2f} nT"
          f"   range {S[m].max()-S[m].min():.1f} nT")
    print()
    print("if the ratio of the third line to the fourth is well under one, beta")
    print("is constant to within the noise -- but see the docstring before")
    print("concluding that it can therefore be frozen, because it cannot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
