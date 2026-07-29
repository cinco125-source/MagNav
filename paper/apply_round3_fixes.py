#!/usr/bin/env python3
"""Apply the round-three review fixes to the manuscript.

The substantive change is the recommended setting. The per-column prior was
recommended on the grounds that it keeps the coefficients physically sized; the
saved estimates say the opposite (median max|beta| of 2485 to 5995 against 787
to 1167 for a shared sigma_beta = 100), so the recommendation moves to
sigma_beta = 100 and the coefficient-magnitude behaviour is reported rather than
claimed. The rest are number and reference corrections.

Run: python paper/apply_round3_fixes.py
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "taes_fgo_magnav.tex")

SUBS = [
    # T1-1: the withdrawn magnitude-admissibility mechanism, in the abstract
    ("""One scalar standard deviation
shared by a regressor whose columns span four orders of magnitude leaves roughly
\\SI{2000}{\\nano\\tesla} of uncompensated cabin field with no nuisance channel
scaled to hold it, and the per-epoch smoother charges it to position through the
map gradient and leaves the map, which no lag, robust kernel, map resolution,
relinearization schedule, measurement rate, or measurement-noise rescaling
repairs; a prior set per regressor column, at \\SI{100}{\\nano\\tesla} of
contribution each, removes it on every affected line.""",
     """One scalar standard deviation
shared by a regressor whose columns span four orders of magnitude leaves the
first linearizations stiff in the compensation relative to position, so roughly
\\SI{2000}{\\nano\\tesla} of uncompensated cabin field is charged to position
through the map gradient and the estimate leaves the map. No lag, robust kernel,
map resolution, relinearization schedule, measurement rate, process-noise floor,
or single choice of measurement-noise scale repairs it; raising that one
standard deviation by two orders of magnitude removes it on every affected
line."""),
    # T1-1: the same mechanism in the body
    ("""A cold start on Mag~4 therefore
presents the graph with roughly \\SI{2000}{\\nano\\tesla} that neither nuisance
channel is scaled to explain, and the one remaining degree of freedom that can
absorb it is position, through the map gradient.""",
     """A cold start on Mag~4 therefore
presents the graph with roughly \\SI{2000}{\\nano\\tesla} while both nuisance
channels are stiff, and the degree of freedom that is not stiff is position,
through the map gradient."""),
    ("""which is consistent with the reading above: any
channel wide enough to hold the platform field will do, and the compensation
basis is the channel that should hold it.""",
     """so the effect is not specific to the compensation
block, although the compensation basis is where the interference belongs."""),
    # T1-2: the recommendation and its justification
    ("""Replacing the
shared scalar with a per-column prior, one that assigns each basis column a
standard deviation of \\SI{100}{\\nano\\tesla} of measurement contribution rather
than a common coefficient scale, does the same at a physically sized setting: the
Mag~4 lines settle at \\SI{19.0}{}, \\SI{25.1}{}, \\SI{98.9}{}, \\SI{30.5}{} and
\\SI{23.7}{\\meter} and the Mag~5 lines at \\SI{12.0}{}, \\SI{9.9}{}, \\SI{102.7}{},
\\SI{16.3}{} and \\SI{11.6}{\\meter}, with the coefficients left near the magnitudes
the physics implies rather than free to grow by three orders of magnitude as they
do under a uniformly loosened prior. That is the setting we recommend, and it is
the one the rest of this subsection uses.""",
     """We recommend
$\\sigma_{\\beta}=100$ and use it for the rest of this subsection: the Mag~4 lines
settle at \\num{19.6}, \\num{25.0}, \\num{99.1}, \\num{28.4} and
\\SI{24.2}{\\meter} and the Mag~5 lines at \\num{12.0}, \\num{10.5}, \\num{101.0},
\\num{15.8} and \\SI{11.5}{\\meter}. A per-column prior, one that assigns each
basis column a standard deviation of \\SI{100}{\\nano\\tesla} of measurement
contribution instead of a common coefficient scale, is the dimensionally
consistent version of the same change and performs the same to within the
line-to-line spread (Table~\\ref{tab:priorsweep}). It does not, however, keep the
coefficients smaller: after warm-up the median of $\\|\\boldsymbol{\\beta}\\|_\\infty$
is \\numrange{787}{1167} across the Mag~4 lines at $\\sigma_{\\beta}=100$ and
\\numrange{2485}{5995} under the per-column prior, against the
\\numrange{552}{722} that a ridge fit of the same data needs. Any prior loose
enough to fix the cold start also lets the compensation basis absorb model error
beyond the platform field, and we report that as a cost of the setting rather
than a property we have controlled."""),
    # T2-5: the FOGM sentence contradicted its own table
    ("""Widening the FOGM prior alone instead,
by a factor of \\numrange{10}{100}, recovers three of the five lines and leaves
1003.02 and 1006.08 diverged,""",
     """Widening the FOGM disturbance prior and
walk instead, leaving the compensation prior nominal, recovers three of the five
lines at a factor of \\num{10} and four at a factor of \\num{100}, leaving 1006.08
diverged in both cases (Table~\\ref{tab:priorsweep}),"""),
    # T2-9: the coefficient range was one line's
    ("""the window solves of Section~\\ref{sec:window1} reach $\\|\\boldsymbol{\\beta}\\|_\\infty$
of \\numrange{544}{795} even at $\\sigma_{\\beta}=1$,""",
     """the window solves of Section~\\ref{sec:window1} reach $\\|\\boldsymbol{\\beta}\\|_\\infty$
of \\numrange{463}{795} across lines at $W\\ge\\SI{100}{\\second}$ even at
$\\sigma_{\\beta}=1$,"""),
    # T2-8: seven alternatives, not six
    ("Six alternatives can be dismissed by measurement",
     "Seven alternatives can be dismissed by measurement"),
    # T1-3: the measurement-noise control, restated to match the log
    ("""setting $\\sqrt{R}$ to \\num{65} and
\\SI{150}{\\nano\\tesla} across the five Mag~4 lines leaves the divergence in place
and makes it erratic rather than smaller, from \\SI{242}{\\meter} to
\\SI{46}{\\kilo\\meter} depending on the line, with no line reaching the tens of
metres that widening the compensation prior reaches on all of them.""",
     """no single value of $\\sqrt{R}$ repairs
it across lines. Setting it to \\SI{150}{\\nano\\tesla} recovers 1007.06 to
\\SI{64.5}{\\meter} and leaves 1003.02 at \\SI{31}{\\kilo\\meter}; setting it to
\\SI{65}{\\nano\\tesla} recovers 1007.02 to \\SI{41.8}{\\meter} and leaves 1007.06
at \\SI{7.3}{\\kilo\\meter}. Across the ten runs the result ranges from
\\SI{42}{\\meter} to \\SI{46}{\\kilo\\meter} with no setting bounded everywhere,
where raising $\\sigma_{\\beta}$ is bounded on every line at once."""),
    # T1-4: timing belongs to the recommended configuration
    ("""Across the ten cases, the update takes
\\SIrange{6.1}{10.4}{\\milli\\second} per state, mean \\SI{8.0}{\\milli\\second}, so an
\\SI{87}{\\minute} line finishes in \\SI{43}{\\second}""",
     """Across the ten cases, the update takes
\\SIrange{6.0}{9.0}{\\milli\\second} per state, mean \\SI{7.8}{\\milli\\second}, so an
\\SI{87}{\\minute} line finishes in \\SI{42}{\\second}"""),
    ("""(\\SIrange{6.1}{10.4}{\\milli\\second} per state at a \\SI{300}{\\second} lag and
        \\SI{1}{\\hertz} states, two orders of magnitude within the state cadence).""",
     """(\\SIrange{6.0}{9.0}{\\milli\\second} per state at a \\SI{300}{\\second} lag and
        \\SI{1}{\\hertz} states, two orders of magnitude within the state cadence)."""),
    ("""runs a \\SI{300}{\\second} lag in real time at
\\SIrange{6.1}{10.4}{\\milli\\second} per state.""",
     """runs a \\SI{300}{\\second} lag in real time at
\\SIrange{6.0}{9.0}{\\milli\\second} per state."""),
    ("smoother that reaches the real-time limit at \\SI{8}{\\milli\\second} per state.",
     "smoother that reaches the real-time limit at \\SI{7.8}{\\milli\\second} per state."),
    # T3: name the runs the worst-case timings come from
    ("""the corresponding worst cases in the same runs are
\\SI{75.5}{\\milli\\second} per state with full relinearization and
\\SIrange{28.7}{35.1}{\\milli\\second} with all ten raw measurements attached.""",
     """the corresponding costs in the
relinearization and dense-measurement runs are \\SI{75.5}{\\milli\\second} per
state with every variable relinearized at every update and
\\SIrange{28.7}{35.1}{\\milli\\second} with all ten raw measurements attached."""),
    # T2-10: the cold-start transient pointer
    ("holds a horizontal DRMS near \\SI{82}{\\meter} through the\nfirst several minutes before locking on (Section~\\ref{sec:results}-G).",
     "holds a horizontal DRMS near \\SI{82}{\\meter} through the\nfirst several minutes before locking on (Fig.~\\ref{fig:warmstart})."),
    # T3: the ridge fit is defined in sec:window1
    ("Seeding the first window's compensation with the ridge fit\nof Section~\\ref{sec:coldprior}, which changes nothing else,",
     "Seeding the first window's compensation with the ridge fit\nof Section~\\ref{sec:window1}, which changes nothing else,"),
    # T2-6: name the handoff the Julia reference uses
    ("""the GTSAM window returns \\SI{12.4}{\\meter} horizontal DRMS against the Julia
smoother's \\SI{13.1}{\\meter} (Table~\\ref{tab:breadth})""",
     """the GTSAM window returns \\SI{12.4}{\\meter} horizontal DRMS against the Julia
smoother's \\SI{13.1}{\\meter} under the marginalized handoff both use
(Table~\\ref{tab:breadth}; the legacy smoothed handoff gives \\SI{13.8}{\\meter})"""),
    # T2-7: boundedness at the nominal prior is implementation dependent
    ("""The window smoother is bounded there on every
counted case, so its numbers are not themselves an artifact of the prior, but we
have not re-run either the window or the causal EKF baselines at a corrected
prior.""",
     """The window smoother is bounded there on every
counted case, so its numbers are not themselves an artifact of the prior, though
the independent GTSAM window diverges on 1007.06 Mag~4 at that same prior
(Section~\\ref{sec:results}-F), so the boundedness is not robust across
implementations. We have not re-run either the window or the causal EKF baselines
at a corrected prior."""),
    # T3: last temporal trace
    ("were produced at the nominal prior,\nbefore this sensitivity was found.",
     "were produced at the nominal prior."),
    # T3: the sigma_S control scales the walk as well
    ("``$\\sigma_S$''\\ leaves the compensation prior at\nnominal and widens the FOGM disturbance prior a hundredfold instead.",
     "``$\\sigma_S$''\\ leaves the compensation prior at\nnominal and widens the FOGM disturbance prior and walk a hundredfold instead."),
    # T3: label the configuration the decimation comparison was run in
    ("""Attaching all ten raw samples to their
nearest kept state instead of one returns \\SI{26.7}{\\meter} against
\\SI{23.9}{\\meter} on line 1007.06 Mag~4 and \\SI{26.2}{\\meter} against
\\SI{19.2}{\\meter} on 1003.02 Mag~4,""",
     """Attaching all ten raw samples to their
nearest kept state instead of one returns \\SI{26.7}{\\meter} against
\\SI{23.9}{\\meter} on line 1007.06 Mag~4 and \\SI{26.2}{\\meter} against
\\SI{19.2}{\\meter} on 1003.02 Mag~4, both pairs run with the ridge-seeded cold
start rather than the widened prior because that is the pair we have,"""),
    # T3: the live TODO
    ("% TODO: add \\cite{trn2025}", "%"),
]

TABLE_OLD = """Line & Mag & INS & nominal & per column \\\\
\\midrule
1003.02 & 4 & 123.8 & div. & \\textbf{19.0} (55.6) \\\\
1003.02 & 5 & 123.8 & 12.6 (24.4) & 12.0 (21.3) \\\\
1003.08 & 4 & 271.9 & div. & \\textbf{25.1} (47.2) \\\\
1003.08 & 5 & 271.9 & 10.9 (22.4) & 9.9 (17.3) \\\\
1006.08 & 4 & 197.7 & div. & \\textbf{98.9} (128.4) \\\\
1006.08 & 5 & 197.7 & 83.9 (52.4) & 102.7 (73.7) \\\\
1007.02 & 4 & 120.7 & 60.0 (327.8) & \\textbf{30.5} (76.9) \\\\
1007.02 & 5 & 120.7 & 14.0 (40.3) & 16.3 (33.9) \\\\
1007.06 & 4 & 317.5 & div. & \\textbf{23.7} (42.7) \\\\
1007.06 & 5 & 317.5 & 11.7 (19.8) & 11.6 (16.1) \\\\"""

TABLE_NEW = """Line & Mag & INS & $\\sigma_{\\beta}=1$ & $\\sigma_{\\beta}=100$ \\\\
\\midrule
1003.02 & 4 & 123.8 & div. & \\textbf{19.6} (58.2) \\\\
1003.02 & 5 & 123.8 & 12.6 (24.4) & 12.0 (21.9) \\\\
1003.08 & 4 & 271.9 & div. & \\textbf{25.0} (48.9) \\\\
1003.08 & 5 & 271.9 & 10.9 (22.4) & 10.5 (19.2) \\\\
1006.08 & 4 & 197.7 & div. & \\textbf{99.1} (134.8) \\\\
1006.08 & 5 & 197.7 & 83.9 (52.4) & 101.0 (80.1) \\\\
1007.02 & 4 & 120.7 & 60.0 (327.8) & \\textbf{28.4} (74.3) \\\\
1007.02 & 5 & 120.7 & 14.0 (40.3) & 15.8 (33.1) \\\\
1007.06 & 4 & 317.5 & div. & \\textbf{24.2} (42.7) \\\\
1007.06 & 5 & 317.5 & 11.7 (19.8) & 11.5 (16.0) \\\\"""

CAPTION_OLD = ("states, at the nominal compensation prior and with the prior scaled per regressor\n"
               "column (\\SI{100}{\\nano\\tesla} of contribution per column). Smoothed DRMS [m]")
CAPTION_NEW = ("states, at the nominal compensation prior and at the recommended one. Smoothed DRMS [m]")


def main():
    src = io.open(TEX, encoding="utf-8").read()
    missed = []
    for old, new in SUBS + [(TABLE_OLD, TABLE_NEW), (CAPTION_OLD, CAPTION_NEW)]:
        if old in src:
            src = src.replace(old, new)
        else:
            missed.append(old)
    io.open(TEX, "w", encoding="utf-8").write(src)
    total = len(SUBS) + 2
    print(f"applied {total - len(missed)}/{total}")
    for m in missed:
        print(f"  NOT FOUND: {' '.join(m.split())[:78]}")


if __name__ == "__main__":
    main()
