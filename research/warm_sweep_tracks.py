#!/usr/bin/env python3
"""Re-score the committed baseline tracks at any DRMS warm-up cut-off.

The question the tables could not answer was whether scoring the whole line,
transient included, changes the ordering: every baseline number in the repo is
computed at warm=600 s, so nobody had measured how the filters fare inside the
window they are excused from. Re-running them needs Julia and the SGL data, but
for line 1007.06 Mag 4 it needs neither -- dump_baselines.jl already wrote the
full 10 Hz trajectories of the INS, the online-TL EKF and the EKF+TL+NN to
baseline_tracks_1007_06_m4.csv, and a trajectory can be scored at any cut-off.

Result on that case (proposed estimator read from
gtsam_poc/gtsam_poc_result_ps100_1007_06_m4.txt, which already reports every
cut-off):

  warm      INS      EKF   EKF+NN   ours causal  ours 300 s
     0   299.06    72.56    54.51        60.96       29.41
   600   317.50    46.67    39.96        42.72       24.19

Including the transient costs the EKF 55%, the NN filter 36%, our causal output
43% and our smoothed output 22%. The filter loses more than the graph does, so
the margin widens rather than narrows: causal against the EKF goes from 0.92 to
0.84 and smoothed from 0.52 to 0.41. Scoring the whole line is both the more
honest convention for a cold-start claim and the more favourable one.

Two cautions. The tracks come from a sigma_beta = 1 run (dump_baselines.jl uses
P0_TL = I) while the proposed estimator is at 100; on this case the two EKF
priors happen to agree (46.67 here against 46.3 in ekf_sig100_results.csv), so
it does not matter, but it would on a line where the prior decides. And the NN
column is one draw of a random initialization: 39.96 at warm=600 sits below the
ten-seed range of 40.2 to 46.7 in nn_seed_stats.csv, so it is a favourable draw
and no ordering against it should rest on a single number.

Validation: at warm=600 this reproduces the committed fgo_breadth_results.csv
entries for the same line and sensor, INS 317.50 against 318.0 and EKF 46.67
against 46.7. The NN does not reproduce, for the reason above.

Usage: python warm_sweep_tracks.py [tracks.csv] [warm ...]
"""
import os
import sys

import numpy as np

R_EARTH = 6378137.0
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "baseline_tracks_1007_06_m4.csv")
# every cut-off the proposed estimator already reports, for the same case
OURS = {0: (60.96, 29.41), 60: (61.31, 29.51), 300: (48.80, 24.92),
        600: (42.72, 24.19)}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].isdigit() else DEFAULT
    warms = [float(a) for a in sys.argv[1:] if a.replace(".", "").isdigit()] \
        or [0, 60, 300, 600]
    d = np.genfromtxt(path, delimiter=",", names=True)
    t = d["t"] - d["t"][0]
    tlat, tlon = d["true_lat"], d["true_lon"]

    def drms(la, lo, warm):
        m = t >= warm
        if not m.any():
            return float("nan")
        dn = (la[m] - tlat[m]) * R_EARTH
        de = (lo[m] - tlon[m]) * R_EARTH * np.cos(tlat[m])
        return float(np.sqrt(np.mean(dn**2 + de**2)))

    cols = [("INS", "ins"), ("EKF", "ekf"), ("EKF+NN", "nn")]
    cols = [(lbl, p) for lbl, p in cols if f"{p}_lat" in d.dtype.names]
    print(f"{os.path.basename(path)}  N={t.size}  {t[-1]/60:.1f} min")
    print()
    hdr = f"{'warm':>6s}" + "".join(f"{lbl:>10s}" for lbl, _ in cols)
    hdr += f"{'ours causal':>13s}{'ours 300 s':>12s}"
    print(hdr)
    for w in warms:
        line = f"{int(w):6d}" + "".join(f"{drms(d[p+'_lat'], d[p+'_lon'], w):10.2f}"
                                        for _, p in cols)
        o = OURS.get(int(w))
        line += (f"{o[0]:13.2f}{o[1]:12.2f}" if o else f"{'-':>13s}{'-':>12s}")
        print(line)

    if 600 in [int(w) for w in warms] and 0 in [int(w) for w in warms]:
        print()
        print("what including the transient costs each estimator:")
        for lbl, p in cols:
            a, b = drms(d[p+"_lat"], d[p+"_lon"], 600), drms(d[p+"_lat"], d[p+"_lon"], 0)
            print(f"  {lbl:10s}{a:8.2f} -> {b:8.2f}   {(b/a-1)*100:+6.1f}%")
        for lbl, i in (("ours causal", 0), ("ours 300 s", 1)):
            a, b = OURS[600][i], OURS[0][i]
            print(f"  {lbl:10s}{a:8.2f} -> {b:8.2f}   {(b/a-1)*100:+6.1f}%")
        ek6, ek0 = drms(d["ekf_lat"], d["ekf_lon"], 600), drms(d["ekf_lat"], d["ekf_lon"], 0)
        print(f"\n  ours/EKF causal   {OURS[600][0]/ek6:.2f} -> {OURS[0][0]/ek0:.2f}")
        print(f"  ours/EKF smoothed {OURS[600][1]/ek6:.2f} -> {OURS[0][1]/ek0:.2f}")
        print("  the filter loses more to the transient than the graph, so scoring")
        print("  the whole line widens the margin instead of narrowing it")


if __name__ == "__main__":
    main()
