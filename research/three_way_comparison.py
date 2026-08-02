#!/usr/bin/env python3
"""EKF against MPF against the factor graph, kept honest about configuration.

The three estimators were never all run at one operating point, and pooling
their numbers as if they had been is how a comparison goes wrong. There are
three configurations in the repository and each answers a different question,
so they are printed separately.

  A  production: sigma_beta = 100, initial position prior 0.1 m, R = 12 nT.
     What the manuscript reports. The MPF was never run here.
  B  the original breadth run: sigma_beta = 1, R = 12 nT, PF resampling
     threshold 0.1. Where the "MPF diverges on every case" claim comes from.
  C  the MPF's own best shot (mpf_breadth_fix.jl): initial position prior
     3.0 m, R swept over 40 and 100 nT, resampling threshold 0.5 with
     roughening -- the settings mpf_cold_R found keep it bounded. Note the
     looser position prior applies to the graph too, which is why the graph's
     numbers here differ from A and why both estimators lose 1007.02.

WHAT THE THREE SAY TOGETHER. Given its own tuning the MPF stops diverging on
six of eight but never becomes competitive: its best case is 40.9 m where the
graph reads 20.2, and its worst is 2898.7. That is not a cold-start failure. On
the compensated stinger, where there is no compensation to estimate at all,
mpf_clean.jl finds the same thing -- the MPF needs log-weights, rare resampling
and roughening to reach 29.4 m on 1007.06 against the EKF's 20.0 and the graph's
16.7, and reads 121.7 m on 1003.08 against 17.7 and 12.5. The particle filter
struggles with this problem itself: a nearly-flat likelihood in a
high-dimensional state, where the weights collapse onto a few particles long
before the map has said anything.

So the MPF belongs in the paper as a diagnosis, not a scalp. It is the estimator
that shows the difficulty is not "be Bayesian" -- it is the most Bayesian of the
three and the weakest.

Usage: python three_way_comparison.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(HERE, "gtsam_poc")
LINES = ["1007.06", "1007.02", "1003.02", "1003.08"]


def load(name, root=HERE):
    p = os.path.join(root, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def ps100(line, mag):
    p = os.path.join(GT, f"gtsam_poc_result_ps100_{line.replace('.', '_')}_m{mag}.txt")
    if not os.path.exists(p):
        return None
    for ln in open(p):
        if ln.startswith("warm=600s"):
            return [float(x.split("=")[1].split()[0]) for x in ln.split("  ") if "drms" in x]
    return None


def f(x, w=7):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return f"{'-':>{w}}"
    if v != v:
        return f"{'-':>{w}}"
    return f"{'div.':>{w}}" if v > 1e4 else f"{v:{w}.1f}"


def main():
    br = {(f"{float(r['line']):.2f}", r["mag"]): r for r in load("fgo_breadth_results.csv")}
    fx = {}
    for r in load("mpf_breadth_fix_results.csv"):
        fx.setdefault((f"{float(r['line']):.2f}", r["mag"]), r)
    ek = {(r["line"], r["mag"]): r["EKF_online"] for r in load("ekf_sig100_results.csv")
          if float(r["sigma_beta"]) == 100}

    print("A. production: sigma_beta=100, init pos 0.1 m, R=12 nT.  DRMS [m]")
    print("   the MPF was never run at this configuration.")
    print(f"   {'case':14s}{'INS':>7s}{'EKF':>8s}{'EKF+NN':>9s}{'FGO caus':>10s}{'FGO 300s':>10s}")
    for l in LINES:
        for m in (4, 5):
            k = (l, f"Mag {m}"); p = ps100(l, m)
            if not p:
                continue
            print(f"   {l+' Mag'+str(m):14s}{f(br[k]['INS'])}{f(ek.get(k),8)}"
                  f"{f(br[k]['EKF_TLNN'],9)}{p[1]:10.1f}{p[0]:10.1f}")

    print()
    print("B. original breadth: sigma_beta=1, R=12 nT, PF threshold 0.1.  DRMS [m]")
    print("   the source of the 'MPF diverges everywhere' line.")
    print(f"   {'case':14s}{'EKF':>8s}{'EKF+NN':>9s}{'MPF+TL':>9s}{'FGO win':>9s}")
    for l in LINES:
        for m in (4, 5):
            k = (l, f"Mag {m}")
            print(f"   {l+' Mag'+str(m):14s}{f(br[k]['EKF_online'],8)}{f(br[k]['EKF_TLNN'],9)}"
                  f"{f(br[k]['MPF_TL'],9)}{f(br[k]['FGO_win'],9)}")

    print()
    print("C. the MPF's own best shot: init pos 3.0 m, R in {40,100}, threshold 0.5")
    print("   + roughening. The looser position prior applies to the graph too.")
    print(f"   {'case':14s}{'MPF best':>10s}{'FGO':>9s}")
    for l in LINES:
        for m in (4, 5):
            r = fx.get((l, f"Mag {m}"))
            if not r:
                continue
            print(f"   {l+' Mag'+str(m):14s}{f(r['MPF_TL_best'],10)}{f(r['FGO'],9)}")

    cl = load("mpf_clean_results.csv")
    if cl:
        print()
        print("D. the compensated stinger, where there is no compensation to estimate")
        print("   at all. If the MPF's trouble were the cold start it would vanish here.")
        print(f"   {'line':10s}{'method':26s}{'DRMS':>9s}")
        for r in cl:
            if "comp" in r["signal"] and "uncomp" not in r["signal"]:
                print(f"   {r['line']:10s}{r['method']:26s}{f(r['DRMS'],9)}")


if __name__ == "__main__":
    main()
