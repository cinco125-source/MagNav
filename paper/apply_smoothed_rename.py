#!/usr/bin/env python3
"""Unify the output terminology: the estimator's two outputs are the causal
estimate and the (fixed-lag) smoothed estimate; "committed" disappears. Also
switches Table IV's bolding to the best overall entry, renames Table III's row
to state the lag, and shortens the affected captions.

Verbs about filters committing states are untouched; only the output name
changes. Run: python paper/apply_smoothed_rename.py
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "taes_incremental.tex")

PAIRS = [
    # IV-C definition
    ("\\item The \\emph{committed} estimate is the value a variable holds when it leaves\nthe lag, that is after it has been revised by every measurement in the following\n$L$ seconds. It is the more accurate of the two and it is available only $L$\nseconds late.",
     "\\item The \\emph{smoothed} estimate is the value a variable holds when it\nleaves the lag, after revision by every measurement in the following $L$\nseconds: the classical fixed-lag smoothed output. It is the more accurate of\nthe two and it is available only $L$ seconds late."),
    # blanket renames (safe: 'committed' as output name only)
    ("committed estimate", "smoothed estimate"),
    ("committed output", "smoothed output"),
    ("committed value", "smoothed value"),
    ("the committed", "the smoothed"),
    ("its committed", "its smoothed"),
    ("Committed:", "Smoothed:"),
    ("committed at a", "smoothed at a"),
    ("causal and committed", "causal and smoothed"),
    ("causal/committed", "causal/smoothed"),
    ("committed, ", "smoothed, "),
    ("committed}", "smoothed}"),
    ("committed (", "smoothed ("),
    ("committed DRMS", "smoothed DRMS"),
    ("lag $L$ [s], committed", "lag $L$ [s], smoothed"),
    # Table III: name the lag on the proposed rows, bold best overall only
    ("\\textbf{Proposed, causal} & \\textbf{42.7} & \\textbf{16.0} & none \\\\\n\\textbf{Proposed, smoothed} & \\textbf{24.2} & \\textbf{11.5} & \\SI{300}{\\second} \\\\",
     "Proposed, causal & 42.7 & 16.0 & none \\\\\n\\textbf{Proposed, \\SI{300}{\\second} lag (smoothed)} & \\textbf{24.2} & \\textbf{11.5} & \\SI{300}{\\second} \\\\"),
    # Table IV: bold the best overall (the smoothed column), unbold the rest
    ("1007.06 & 4 & 46.7 & 48.8 & \\textbf{42.7} & 24.2 \\\\",
     "1007.06 & 4 & 46.7 & 48.8 & 42.7 & \\textbf{24.2} \\\\"),
    ("1007.06 & 5 & 17.8 & 18.3 & \\textbf{16.0} & 11.5 \\\\",
     "1007.06 & 5 & 17.8 & 18.3 & 16.0 & \\textbf{11.5} \\\\"),
    ("1007.02 & 4 & div. & 130.0 & \\textbf{74.3} & 28.4 \\\\",
     "1007.02 & 4 & div. & 130.0 & 74.3 & \\textbf{28.4} \\\\"),
    ("1007.02 & 5 & 31.6 & \\textbf{30.4} & 33.1 & 15.8 \\\\",
     "1007.02 & 5 & 31.6 & 30.4 & 33.1 & \\textbf{15.8} \\\\"),
    ("1003.02 & 4 & div. & 101.0 & \\textbf{58.2} & 19.6 \\\\",
     "1003.02 & 4 & div. & 101.0 & 58.2 & \\textbf{19.6} \\\\"),
    ("1003.02 & 5 & 28.1 & 29.5 & \\textbf{21.9} & 12.0 \\\\",
     "1003.02 & 5 & 28.1 & 29.5 & 21.9 & \\textbf{12.0} \\\\"),
    ("1003.08 & 4 & err. & \\textbf{46.6} & 48.9 & 25.0 \\\\",
     "1003.08 & 4 & err. & 46.6 & 48.9 & \\textbf{25.0} \\\\"),
    ("1003.08 & 5 & 21.1 & 20.7 & \\textbf{19.2} & 10.5 \\\\",
     "1003.08 & 5 & 21.1 & 20.7 & 19.2 & \\textbf{10.5} \\\\"),
    ("``err.''\\ leaves the map; bold = best causal entry.",
     "``err.''\\ leaves the map; bold = best overall."),
    ("Line & Mag & online & TL{+}NN & causal & \\SI{300}{\\second} \\\\",
     "Line & Mag & online & TL{+}NN & causal & \\SI{300}{\\second} lag \\\\"),
]


def main():
    src = io.open(TEX, encoding="utf-8").read()
    # order matters: run the multi-line specific ones first (already first)
    applied = 0
    for old, new in PAIRS:
        n = src.count(old)
        if n:
            src = src.replace(old, new)
            applied += 1
        else:
            print(f"  NOT FOUND: {' '.join(old.split())[:64]}")
    left = src.count("committed")
    io.open(TEX, "w", encoding="utf-8").write(src)
    print(f"applied {applied}/{len(PAIRS)}; residual 'committed': {left}")


if __name__ == "__main__":
    main()
