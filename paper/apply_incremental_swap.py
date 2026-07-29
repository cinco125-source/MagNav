#!/usr/bin/env python3
"""Correct the causal-column bookkeeping after the FGO column was switched from
the sliding window to the per-epoch smoother.

Read causally the per-epoch smoother is the best of the three causal estimators
on six of the eight counted cases, not seven: it also trails the EKF+TL+NN on
1003.08 Mag 4 (48.9 against 46.6 m).

Run: python paper/apply_incremental_swap.py
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "taes_fgo_magnav.tex")

SUBS = [
    (r"1003.08 & SV & R & 0.4 & 4 & err. & 46.6 & \textbf{48.9} & \textbf{25.0} & 28.0 \\",
     r"1003.08 & SV & R & 0.4 & 4 & err. & \textbf{46.6} & 48.9 & \textbf{25.0} & 28.0 \\"),
    (r"\multicolumn{5}{l}{best of the causal estimators} & & & \textbf{7 / 8} & & \\",
     r"\multicolumn{5}{l}{best of the causal estimators} & & & \textbf{6 / 8} & & \\"),
    ("beating the NN-augmented filter by\n"
     "\\SIrange{7}{102}{\\meter} (Fig.~\\ref{fig:breadth}); read causally it is still ahead\n"
     "on seven of the eight, the exception being 1007.02 Mag~5 at \\SI{33.1}{} against\n"
     "\\SI{30.4}{\\meter};",
     "beating the NN-augmented filter by\n"
     "\\SIrange{6.8}{101.6}{\\meter} (Fig.~\\ref{fig:breadth}); read causally, with no\n"
     "look-ahead at all, it is still ahead on six of the eight, the exceptions being\n"
     "1007.02 Mag~5 (\\num{33.1} against \\SI{30.4}{\\meter}) and 1003.08 Mag~4\n"
     "(\\num{48.9} against \\SI{46.6}{\\meter});"),
]


def main():
    src = io.open(TEX, encoding="utf-8").read()
    missed = []
    for old, new in SUBS:
        if old in src:
            src = src.replace(old, new)
        else:
            missed.append(old)
    io.open(TEX, "w", encoding="utf-8").write(src)
    print(f"applied {len(SUBS) - len(missed)}/{len(SUBS)}")
    for m in missed:
        print(f"  NOT FOUND: {' '.join(m.split())[:76]}")


if __name__ == "__main__":
    main()
