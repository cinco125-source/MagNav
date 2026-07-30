#!/usr/bin/env python3
"""Distribute the tables to their subsections and shorten every caption.

The assembly appended all six tables after the results prose, so LaTeX piles
them at the end of the section. Each table moves to just after the heading of
the subsection that discusses it, and the long captions shrink to one or two
lines; the detail they carried already lives in the body text.

Run: python paper/apply_table_layout.py
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "taes_incremental.tex")

CAPTIONS = {
    "tab:params": "Estimator settings, common to every line and both "
                  "magnetometers.",
    "tab:lines": "Flight lines. FF = free flight, SV = survey, CAL = "
                 "calibration; maps R = Renfrew, E = Eastern; INS = "
                 "free-inertial drift [m].",
    "tab:1007": "Line 1007.06 cold start: horizontal DRMS [m] after a "
                "\\SI{600}{\\second} warm-up. The causal output uses no "
                "look-ahead; the committed output lags by \\SI{300}{\\second}.",
    "tab:breadth": "Breadth: DRMS [m] after a \\SI{600}{\\second} warm-up. "
                   "``div.''\\ $>\\SI{10}{\\kilo\\meter}$, ``err.''\\ leaves "
                   "the map; bold = best causal entry. Line 1006.08 is listed "
                   "but not counted (see text).",
    "tab:lag": "Lag sweep: committed DRMS [m], causal at "
               "$L=\\SI{300}{\\second}$. The causal output varies by under a "
               "metre across the sweep.",
    "tab:ablation": "Ablation on line 1007.06, one change per row: committed "
                    "(causal) DRMS [m].",
}

DEST = {
    "tab:params": r"\subsection{Dataset, maps, and protocol}\label{sec:protocol}",
    "tab:lines": r"\subsection{Dataset, maps, and protocol}\label{sec:protocol}",
    "tab:1007": r"\subsection{Representative line}\label{sec:repline}",
    "tab:breadth": r"\subsection{Breadth}\label{sec:breadth}",
    "tab:lag": r"\subsection{Lag design}\label{sec:lagdesign}",
    "tab:ablation": r"\subsection{Ablation}\label{sec:ablation}",
}


def main():
    src = io.open(TEX, encoding="utf-8").read()

    # extract each table environment by its label
    blocks = {}
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", src, re.S):
        lm = re.search(r"\\label\{(tab:[^}]*)\}", m.group(0))
        if lm and lm.group(1) in CAPTIONS:
            blocks[lm.group(1)] = m.group(0)
    missing = set(CAPTIONS) - set(blocks)
    if missing:
        raise SystemExit(f"tables not found: {missing}")

    # remove them (and the %% Table comment lines around the block dump)
    for b in blocks.values():
        src = src.replace(b, "")
    src = re.sub(r"%% -+ Table [IVX]+\n", "", src)
    src = re.sub(r"%% Section V tables\..*?(?=\\)", "", src, flags=re.S)
    src = re.sub(r"\n{3,}", "\n\n", src)

    # shorten captions and reinsert after the destination heading
    order = ["tab:params", "tab:lines", "tab:1007", "tab:breadth",
             "tab:lag", "tab:ablation"]
    for lab in order:
        block = blocks[lab]
        block = re.sub(r"\\caption\{.*?\}\n\\label",
                       "\\\\caption{" + CAPTIONS[lab].replace("\\", "\\\\")
                       + "}\n\\\\label", block, count=1, flags=re.S)
        anchor = DEST[lab]
        idx = src.index(anchor)
        # params goes before lines at the same anchor: insert after anchor line,
        # later inserts push earlier ones down, so insert in reverse order there
        insert_at = idx + len(anchor)
        src = src[:insert_at] + "\n\n" + block + "\n" + src[insert_at:]
    io.open(TEX, "w", encoding="utf-8").write(src)
    print("tables placed and captions shortened")


if __name__ == "__main__":
    main()
