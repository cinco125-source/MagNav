#!/usr/bin/env python3
"""Remove line 1006.08 from the tables and the body text.

The calibration line was listed but never counted; it now leaves the paper
entirely. The lines table, the breadth table, the protocol prose and the
breadth prose lose their 1006.08 content, and the caption note about it goes.

Run: python paper/apply_drop_1006.py
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "taes_incremental.tex")

SUBS = [
    # lines table row
    ("1006.08 & CAL & E & 398 & 14.0 & 197.7 \\\\\n", ""),
    # lines table caption loses CAL
    ("FF = free flight, SV = survey, CAL = calibration; maps",
     "FF = free flight, SV = survey; maps"),
    # breadth table: the set-aside block and its rules
    ("\\midrule\n1006.08 & 4 & div. & 1173.7 & --- & 134.8 & 99.1 \\\\\n"
     "1006.08 & 5 & 117.5 & 105.2 & 793.7 & 80.1 & 101.0 \\\\\n", ""),
    ("``err.''\\ leaves the map; bold = best causal entry. Line 1006.08 is "
     "listed but not counted (see text).",
     "``err.''\\ leaves the map; bold = best causal entry."),
    ("\\multicolumn{5}{l}{best causal, counted cases}",
     "\\multicolumn{5}{l}{best causal}"),
    ("\\multicolumn{5}{l}{best overall, counted cases}",
     "\\multicolumn{5}{l}{best overall}"),
    # protocol prose
    ("Flt1006 is a calibration-man\u0153uvre flight; its single short line is "
     "listed for\ncompleteness but not counted, since at \\SI{14}{\\minute} "
     "the warm-up leaves too\nlittle line to score. The map-building flights "
     "(Flt1004--1005) are not used.",
     "Flt1006 is a calibration-man\u0153uvre flight and, with the map-building "
     "flights\n(Flt1004--1005), is not used: its single \\SI{14}{\\minute} "
     "line leaves too little\nto score after the warm-up."),
    # breadth prose
    ("\nThe set-aside calibration line 1006.08 behaves as expected from its "
     "length and\ncontent, and no conclusion is drawn from it.", ""),
    # representative-line prose count references stay valid (counted cases)
]


def main():
    src = io.open(TEX, encoding="utf-8").read()
    applied = 0
    for old, new in SUBS:
        if old in src:
            src = src.replace(old, new)
            applied += 1
        else:
            # tolerate oe-ligature vs plain spelling
            alt = old.replace("\u0153", "oe")
            if alt in src:
                src = src.replace(alt, new.replace("\u0153", "oe"))
                applied += 1
            else:
                print(f"  NOT FOUND: {' '.join(old.split())[:64]}")
    # any straggler mentions
    n = len(re.findall(r"1006\.08", src))
    io.open(TEX, "w", encoding="utf-8").write(src)
    print(f"applied {applied}/{len(SUBS)}; remaining 1006.08 mentions: {n}")


if __name__ == "__main__":
    main()
