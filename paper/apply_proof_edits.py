#!/usr/bin/env python3
"""Apply the proofreading pass to the cold-start sections.

Each entry is (current text, replacement). The script reports any entry it
cannot find so a silently skipped edit is never mistaken for an applied one.

Run: python paper/apply_proof_edits.py
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "taes_fgo_magnav.tex")

SUBS = [
    ("two orders of magnitude inside the state cadence",
     "two orders of magnitude within the state cadence"),
    ("can be told apart from", "can be distinguished from"),
    ("here the compensation can reach the interference and is merely slow, there\nit cannot reach it at all.",
     "in the first regime the compensation can reach the interference and is merely\nslow; in the second it cannot reach it at all."),
    ("at the identical inputs and\nthe identical prior",
     "with identical inputs and the\nidentical prior"),
    ("Taken to its per-epoch limit the smoother exposes",
     "Taken to its per-epoch limit, the smoother exposes"),
    ("Over the first\n\\SI{300}{\\second} the residual",
     "Over the first\n\\SI{300}{\\second}, the residual"),
    ("The asymmetry is the clue, and it is a magnitude asymmetry.",
     "The asymmetry is the clue, and it is one of magnitude."),
    ("the prior the whole comparison in this paper has used",
     "the prior this paper's whole comparison has used"),
    ("while they are still travelling", "while they are still traveling"),
    ("The explanation is testable one window at a time",
     "The explanation can be tested one window at a time"),
    ("At $\\sigma_{\\beta}=1$ the Mag~4 solutions",
     "At $\\sigma_{\\beta}=1$, the Mag~4 solutions"),
    ("At\n$\\sigma_{\\beta}=1000$ that region is gone",
     "At\n$\\sigma_{\\beta}=1000$, that region is gone"),
    ("Across the \\num{78} line, magnetometer and $W$ combinations",
     "Across the \\num{78} combinations of line, magnetometer, and $W$"),
    ("in only \\num{7}; of those the seeded start",
     "in only \\num{7}; of those, the seeded start"),
    ("One negative result belongs here because we went looking for it.",
     "One negative result belongs here because we specifically tested for it."),
    ("Across the ten cases the update takes",
     "Across the ten cases, the update takes"),
    ("at the full \\SI{10}{\\hertz} state cadence that lag is a window ten times as large and remains\noffline",
     "at the full \\SI{10}{\\hertz} state cadence, that lag is a window ten times as large and\nremains offline"),
    ("must admit the platform field the installation actually\ncarries",
     "must admit the platform field that the installation actually\ncarries"),
    ("an entry below about \\num{100} is at or under the drift it has to beat",
     "an entry below about \\SI{100}{\\meter} is at or under the drift it has to beat"),
]


def main():
    src = io.open(TEX, encoding="utf-8").read()
    missed = []
    for old, new in SUBS:
        if old in src:
            src = src.replace(old, new)
        else:
            missed.append(old)
    # \SI{x}{} with an empty unit is a siunitx misuse; \num is the right macro
    src, n_si = re.subn(r"\\SI\{([0-9.]+)\}\{\}", r"\\num{\1}", src)
    io.open(TEX, "w", encoding="utf-8").write(src)
    print(f"applied {len(SUBS) - len(missed)}/{len(SUBS)} edits, "
          f"{n_si} empty-unit \\SI converted to \\num")
    for m in missed:
        print(f"  NOT FOUND: {m[:70]}")


if __name__ == "__main__":
    main()
