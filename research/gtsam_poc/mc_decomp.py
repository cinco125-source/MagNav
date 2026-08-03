#!/usr/bin/env python3
"""Where the margin over the EKF actually comes from, one source at a time.

The comparison is run with NO robust kernel on either side. That is what
src/fgo.jl does by default (robust = :none) and what src/ekf_online.jl does
always -- it has no kernel, no innovation gating and no outlier rejection
anywhere. Turning Huber on from the first linearization was something the GTSAM
runner introduced, and every FGO-against-EKF number produced that way was a
comparison between an estimator holding an M-estimator and one that was not.
The kernel-on runs are kept here as one row of the decomposition rather than as
the headline.

Three things could be producing the graph's margin, and each has a control that
removes exactly one of them while the seeds, the data and the perturbation stay
identical:

  robust kernel     --norobust on the graph; and the same Huber given to the
                    filter, which answers it from the other side
  relinearization   --norelin freezes every linearization where a filter would
                    put it, leaving linear fixed-lag smoothing
  smoothing         the causal column already excludes it; the 300 s column is
                    what it buys

Run mc_1007.py at scales 0.1 and 3.0 in four configurations (default,
--norobust, --norelin, --norelin --norobust), then this.

WHAT IT CAME TO, 1007.06 segment, Mag 5, sigma_beta = 100, whole segment scored:

The kernel is not what the graph's margin is made of. Taking it off the graph
moves the causal ratio from 0.969 to 0.970 at a realistic initial error and
from 0.877 to 0.896 at three times the prior -- two percent at the outside.
Given to the FILTER instead it is worth 0.977 at scale 0.1 (30 seeds of 30) and
exactly nothing at scale 3.0 (0.999, 12 of 30). So an early single-seed reading
in which the graph's 0.977 and the kernel's 0.977 coincided was a coincidence,
not a cause.

What is left, kernel-free on both sides:

  scale 0.1   causal 0.970 (30/30)   smoothed 0.250 (30/30)
  scale 3.0   causal 0.896 (27/30)   smoothed 0.268 (27/30)

and the causal three percent at scale 0.1 is nearly all smoothing-adjacent
rather than the formulation: the norelin control keeps 0.991 of it. At scale
3.0 the norelin control gives back 12% with 26 seeds of 30 at p = 0.0001, which
is the one place the formulation itself is doing the work.

READ IT AS: the smoothed output is worth a factor of four and the graph is not
needed to explain it -- a linear fixed-lag smoother on the same model gets
essentially the same thing. The causal output is worth one to three percent at
a realistic start, which is not a contribution in any framing. The formulation
earns its keep only when the initial error is several times the assumed prior,
and it earns about 12% there, with a tail that runs the wrong way on a few
starts.
"""
import csv
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# tag -> what has been removed from the graph
CONFIG = {
    "k": "relin + huber      (as the GTSAM runner shipped)",
    "nk": "relin, NO huber    (matches fgo.jl's default)",
    "nr": "NO relin + huber   (linear fixed-lag smoothing)",
    "nrnk": "NO relin, NO huber (the same, kernel-free)",
}
SCALES = ("0.1", "3.0")


def sign_test(r):
    n = r.size
    w = int((r < 1).sum())
    k = min(w, n - w)
    return w, min(2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n, 1.0)


def load(tag, scale):
    p = os.path.join(HERE, f"mc_1007_{tag}_s{scale}.csv")
    if not os.path.exists(p):
        return None
    return {int(x["seed"]): x for x in csv.DictReader(open(p))}


def col(d, name):
    return np.array([float(d[k][name]) for k in sorted(d)])


def main():
    print("initial-error MC, 1007.06 segment, whole segment scored (warm = 0), 30 seeds")
    print("ratios are against the PLAIN EKF -- no kernel, no gating, as ekf_online.jl is")
    print()
    for s in SCALES:
        print(f"--- scale {s} " + "-" * 52)
        base = None
        for tag, what in CONFIG.items():
            d = load(tag, s)
            if d is None:
                print(f"  {what:36s}  (not run)")
                continue
            ek = col(d, "EKF")
            if base is None:
                base = d
                eh = col(d, "EKF_huber")
                wh, _ = sign_test(eh / ek)
                print(f"  {'EKF':36s}{np.median(ek):9.1f} m")
                print(f"  {'EKF + huber (the kernel alone)':36s}{np.median(eh):9.1f} m"
                      f"{math.exp(np.log(eh/ek).mean()):8.3f}{wh:>4}/{ek.size}")
            rc = col(d, "FGO_causal") / ek
            rs = col(d, "FGO_smoothed") / ek
            wc, _ = sign_test(rc)
            ws, _ = sign_test(rs)
            n = ek.size
            print(f"  {what:36s}{np.median(col(d,'FGO_causal')):9.1f} m"
                  f"{math.exp(np.log(rc).mean()):8.3f}{wc:>4}/{n}"
                  f"   |  smoothed{math.exp(np.log(rs).mean()):8.3f}{ws:>4}/{n}")
        # what each knob is worth, paired seed by seed at fixed kernel setting
        print()
        for a_tag, b_tag, label in (("nk", "nrnk", "relinearization, kernel-free"),
                                    ("k", "nr", "relinearization, kernel on"),
                                    ("k", "nk", "the kernel, on the graph")):
            a, b = load(a_tag, s), load(b_tag, s)
            if not a or not b:
                continue
            ks = sorted(set(a) & set(b))
            for name, lbl in (("FGO_causal", "causal"), ("FGO_smoothed", "smoothed")):
                x = np.array([float(a[k][name]) for k in ks])
                y = np.array([float(b[k][name]) for k in ks])
                r = x / y
                w, p = sign_test(r)
                print(f"    {label:30s} {lbl:9s}{math.exp(np.log(r).mean()):8.3f}"
                      f"{w:>4}/{len(ks)}  p={p:.4f}  worst {r.max():.2f}")
        print()


if __name__ == "__main__":
    main()
