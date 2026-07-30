#!/usr/bin/env python3
"""Second pass of the terminology unification, against the actual table rows
(the breadth table still carries the MPF column) and the four residual
'committed' occurrences. Run: python paper/apply_smoothed_rename2.py"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "taes_incremental.tex")

PAIRS = [
    # residual prose
    ("emits a causal answer now and a committed answer revised by",
     "emits a causal answer now and a smoothed answer revised by"),
    ("revise a committed state, the estimate can diverge.",
     "revise a state once made, the estimate can diverge."),
    ("committed one revised by the full lag, and comparisons in the",
     "smoothed one revised by the full lag, and comparisons in the"),
    ("The committed trace is the", "The smoothed trace is the"),
    # Table IV rows (7 columns, with MPF)
    ("1007.06 & 4 & 46.7 & 48.8 & 1227.0 & \\textbf{42.7} & 24.2 \\\\",
     "1007.06 & 4 & 46.7 & 48.8 & 1227.0 & 42.7 & \\textbf{24.2} \\\\"),
    ("1007.06 & 5 & 17.8 & 18.3 & 75.1 & \\textbf{16.0} & 11.5 \\\\",
     "1007.06 & 5 & 17.8 & 18.3 & 75.1 & 16.0 & \\textbf{11.5} \\\\"),
    ("1007.02 & 4 & div. & 130.0 & 1311.5 & \\textbf{74.3} & 28.4 \\\\",
     "1007.02 & 4 & div. & 130.0 & 1311.5 & 74.3 & \\textbf{28.4} \\\\"),
    ("1007.02 & 5 & 31.6 & \\textbf{30.4} & 2247.3 & 33.1 & 15.8 \\\\",
     "1007.02 & 5 & 31.6 & 30.4 & 2247.3 & 33.1 & \\textbf{15.8} \\\\"),
    ("1003.02 & 4 & div. & 101.0 & 482.6 & \\textbf{58.2} & 19.6 \\\\",
     "1003.02 & 4 & div. & 101.0 & 482.6 & 58.2 & \\textbf{19.6} \\\\"),
    ("1003.02 & 5 & 28.1 & 29.5 & 40.9 & \\textbf{21.9} & 12.0 \\\\",
     "1003.02 & 5 & 28.1 & 29.5 & 40.9 & 21.9 & \\textbf{12.0} \\\\"),
    ("1003.08 & 4 & err. & \\textbf{46.6} & 2898.7 & 48.9 & 25.0 \\\\",
     "1003.08 & 4 & err. & 46.6 & 2898.7 & 48.9 & \\textbf{25.0} \\\\"),
    ("1003.08 & 5 & 21.1 & 20.7 & 77.1 & \\textbf{19.2} & 10.5 \\\\",
     "1003.08 & 5 & 21.1 & 20.7 & 77.1 & 19.2 & \\textbf{10.5} \\\\"),
    ("Line & Mag & online & TL{+}NN & TL & causal & \\SI{300}{\\second} \\\\",
     "Line & Mag & online & TL{+}NN & TL & causal & \\SI{300}{\\second} lag \\\\"),
    ("bold = best causal entry.", "bold = best overall."),
    ("bold = best causal\nestimator in each row.", "bold = best overall in\neach row."),
]


def main():
    src = io.open(TEX, encoding="utf-8").read()
    applied = 0
    for old, new in PAIRS:
        if old in src:
            src = src.replace(old, new)
            applied += 1
        else:
            print(f"  NOT FOUND: {' '.join(old.split())[:66]}")
    print(f"applied {applied}/{len(PAIRS)}; residual 'committed': "
          f"{src.count('committed')}")
    io.open(TEX, "w", encoding="utf-8").write(src)


if __name__ == "__main__":
    main()
