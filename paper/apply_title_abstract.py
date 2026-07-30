#!/usr/bin/env python3
"""Set the new manuscript's title and abstract.

The title leads with the joint estimation, per the agreed direction; the
abstract is rewritten around the single proposed estimator, the
causal/committed protocol, and the flight-data numbers of Tables III-VI.

Run: python paper/apply_title_abstract.py
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "taes_incremental.tex")

TITLE = (r"\title{Joint Navigation and Online Aeromagnetic Compensation by "
         r"Incremental Factor-Graph Smoothing}")

MARK = (r"\markboth{LEE ET AL.}{JOINT NAVIGATION AND ONLINE AEROMAGNETIC "
        r"COMPENSATION BY INCREMENTAL FACTOR-GRAPH SMOOTHING}")

ABSTRACT = r"""\begin{abstract}
Airborne magnetic-anomaly navigation bounds inertial drift in GNSS-denied
flight by matching a scalar magnetometer against an anomaly map. Its central
obstacle is the aircraft's own magnetic field, which must be compensated online
when no calibration flight is available, the cold-start regime; on a
cabin-mounted magnetometer the uncompensated field exceeds the full range of
the map signal itself. We cast navigation and compensation as one
maximum-a-posteriori problem on a factor graph, carrying the Tolles--Lawson
coefficients as time-varying variables alongside the inertial error state, and
solve it on board with an incremental fixed-lag smoother built on iSAM2, at a
\SI{300}{\second} lag and \SI{1}{\hertz} states,
\SIrange{6.0}{9.0}{\milli\second} per update. The smoother produces two
estimates and we report both throughout: a causal estimate with no look-ahead,
which is the like-for-like comparison against recursive filters, and a
committed estimate revised by the full lag, which is the output of a system
permitted to lag real time. On the SGL~2020 flight data, across eight
cold-start cases spanning four lines, two flights, two maps and two altitude
regimes under one untuned configuration, the causal estimate alone beats an
identically modelled online-TL EKF and a neural-network-augmented EKF on six of
eight cases, and the committed estimate is the most accurate on all eight, at
\SIrange{10.5}{28.4}{\meter} where free-inertial drift is hundreds of metres. A
lag sweep locates the design knee near \SI{300}{\second} and shows the causal
estimate is indifferent to the lag, so look-ahead buys smoothing accuracy only;
an ablation over every element of the configuration finds metre-level
sensitivity and no delicate setting.
\end{abstract}"""


def main():
    src = io.open(TEX, encoding="utf-8").read()
    src = re.sub(r"\\title\{[^}]*\}", TITLE.replace("\\", "\\\\"), src, count=1)
    src = re.sub(r"\\markboth\{[^}]*\}\{[^}]*\}", MARK.replace("\\", "\\\\"),
                 src, count=1)
    src = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}",
                 ABSTRACT.replace("\\", "\\\\"), src, count=1, flags=re.S)
    io.open(TEX, "w", encoding="utf-8").write(src)
    print("title, running head and abstract replaced")


if __name__ == "__main__":
    main()
