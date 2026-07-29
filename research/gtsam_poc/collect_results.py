#!/usr/bin/env python3
"""Collect the incremental-smoother sweeps into one table.

Columns: the Julia sliding-window reference (research/fgo_breadth_results.csv),
the per-epoch incremental smoother with no bootstrap, with the Tolles-Lawson
prior bootstrap only, and with the batch bootstrap.

Usage: collect_results.py [warm=600]
"""
import csv
import os
import re
import sys

G = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(G))
LINES = ["1003_02", "1003_08", "1006_08", "1007_02", "1007_06"]
VARIANTS = [("none", "_dec300_{L}_m{M}"),
            ("TL only", "_dec300tl_{L}_m{M}"),
            ("batch900 only", "_dec300boot_{L}_m{M}"),
            ("TL+batch300", "_dec300br_{L}_m{M}")]


def read_result(tag, warm="600"):
    p = os.path.join(G, f"gtsam_poc_result{tag}.txt")
    if not os.path.exists(p):
        return None, None, None
    sm = rt = ins = None
    for line in open(p):
        m = re.match(rf"warm={warm}s\s+smoothed_drms=([\d.]+) m\s+"
                     rf"realtime_drms=([\d.]+) m\s+ins_drms=([\d.]+) m", line)
        if m:
            sm, rt, ins = (float(x) for x in m.groups())
    return sm, rt, ins


def julia_ref():
    ref = {}
    p = os.path.join(ROOT, "research", "fgo_breadth_results.csv")
    with open(p) as fh:
        for row in csv.DictReader(fh):
            key = (row["line"].replace(".", "_"), row["mag"][-1])
            ref[key] = (float(row["INS"]), row["EKF_online"], row["EKF_TLNN"],
                        row["FGO_win"])
    return ref


def main():
    warm = sys.argv[1] if len(sys.argv) > 1 else "600"
    ref = julia_ref()
    print(f"horizontal DRMS [m], warm-up {warm} s, smoothed (realtime)\n")
    hdr = (f"{'line':9s} {'mag':4s} {'INS':>8s} {'EKF-onl':>10s} {'EKF+NN':>9s} "
           f"{'Julia win':>10s} " + " ".join(f"{n:>18s}" for n, _ in VARIANTS))
    print(hdr)
    print("-" * len(hdr))
    for L in LINES:
        for M in ("4", "5"):
            r = ref.get((L, M))
            cols = []
            for _, pat in VARIANTS:
                sm, rt, _ = read_result(pat.format(L=L, M=M), warm)
                cols.append("-" if sm is None else f"{sm:8.1f} ({rt:6.1f})")
            ins = f"{r[0]:8.1f}" if r else " " * 8
            eko = f"{float(r[1]):10.1f}" if r and r[1] not in ("NaN", "Inf") else f"{'--':>10s}"
            enn = f"{float(r[2]):9.1f}" if r and r[2] not in ("NaN", "Inf") else f"{'--':>9s}"
            fgw = f"{float(r[3]):10.1f}" if r and r[3] not in ("NaN", "Inf") else f"{'--':>10s}"
            print(f"{L.replace('_', '.'):9s} {'Mag ' + M:4s} {ins} {eko} {enn} "
                  f"{fgw} " + " ".join(f"{c:>18s}" for c in cols))


if __name__ == "__main__":
    main()
