#!/usr/bin/env python3
"""Compare the cold-started and ridge-seeded window solves at equal W.

For each case and window length prints the final objective and position error of
both starts, and flags whether picking the LOWER objective would also have picked
the more accurate solution. That is the claim that makes the failure detectable
online: the estimator can compare two costs it can compute, without truth.
"""
import os
import re
import sys

G = os.path.dirname(os.path.abspath(__file__))


def parse(path):
    out = {}
    line = mag = None
    for ln in open(path):
        m = re.match(r"#+ (\S+) #+", ln)
        if m:
            line = m.group(1).replace("_", ".")
        m = re.match(r"mag(\d) K=", ln)
        if m:
            mag = m.group(1)
        m = re.match(r"W=\s*(\d+).*->\s+([\d.e+]+)\s+drms\s+([\d.]+) m", ln)
        if m:
            out.setdefault((line, mag), {})[int(m.group(1))] = (
                float(m.group(2)), float(m.group(3)))
    return out


def main():
    cold = parse(os.path.join(G, "knee_sweep.log"))
    ridge = parse(os.path.join(G, "knee_ridge_causal.log"))
    print(f"{'line':9s}{'mag':4s}{'W':>6s}"
          f"{'err cold':>13s}{'err ridge':>13s}"
          f"{'drms cold':>11s}{'drms ridge':>11s}  pick-by-cost")
    agree = total = 0
    for key in sorted(set(cold) & set(ridge)):
        for W in sorted(set(cold[key]) & set(ridge[key])):
            ec, dc = cold[key][W]
            er, dr = ridge[key][W]
            if abs(ec - er) / max(ec, er) < 1e-4:
                verdict = "tie"
            else:
                cheaper_is_better = (ec < er) == (dc < dr)
                verdict = "ok" if cheaper_is_better else "WRONG"
                total += 1
                agree += cheaper_is_better
            print(f"{key[0]:9s}{'Mag'+key[1]:4s}{W:6d}"
                  f"{ec:13.4e}{er:13.4e}{dc:11.1f}{dr:11.1f}  {verdict}")
    if total:
        print(f"\ncost-based selection picks the more accurate solution in "
              f"{agree}/{total} cases where the two differ")


if __name__ == "__main__":
    main()
