#!/usr/bin/env python3
"""Wire the new figures into the manuscript with short captions.

fig_concept's caption is rewritten to match its new content; fig_track and
fig_comp join the representative-line subsection, fig_breadthbar the breadth
subsection, fig_lagsweep the lag-design subsection, each with a one-or-two-line
caption and a prose pointer.

Run: python paper/apply_wire_figures.py
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "taes_incremental.tex")

CONCEPT_CAP = (
    "\\caption{The cold-start problem on line 1007.06. (a)~Anomaly map and the "
    "flown track. (b)~Measured scalar minus the map value along the track (the "
    "uncompensated aircraft field, Mag~4) against the map signal itself: the "
    "interference exceeds the signal's full range.}")

FIGS = {
    "fig:track": (
        r"\subsection{Representative line}\label{sec:repline}",
        "\\begin{figure}[t]\n\\centering\n"
        "\\includegraphics[width=\\columnwidth]{figs/fig_track.pdf}\n"
        "\\caption{Line 1007.06: committed estimate against truth on the "
        "anomaly map.}\n"
        "\\label{fig:track}\n\\end{figure}\n"),
    "fig:comp": (
        r"\subsection{Representative line}\label{sec:repline}",
        "\\begin{figure}[t]\n\\centering\n"
        "\\includegraphics[width=\\columnwidth]{figs/fig_comp.pdf}\n"
        "\\caption{What the compensation chain learned (Mag~4): the estimated "
        "$\\mathbf{A}^{\\top}\\boldsymbol{\\beta}+S$ overlays the interference "
        "reference; the residual is tens of nanotesla.}\n"
        "\\label{fig:comp}\n\\end{figure}\n"),
    "fig:breadthbar": (
        r"\subsection{Breadth}\label{sec:breadth}",
        "\\begin{figure}[t]\n\\centering\n"
        "\\includegraphics[width=\\columnwidth]{figs/fig_breadthbar.pdf}\n"
        "\\caption{Breadth comparison of Table~\\ref{tab:breadth} on a log "
        "axis.}\n"
        "\\label{fig:breadthbar}\n\\end{figure}\n"),
    "fig:lagsweep": (
        r"\subsection{Lag design}\label{sec:lagdesign}",
        "\\begin{figure}[t]\n\\centering\n"
        "\\includegraphics[width=\\columnwidth]{figs/fig_lagsweep.pdf}\n"
        "\\caption{Committed DRMS against lag (Table~\\ref{tab:lag}); open "
        "markers show the lag-independent causal error.}\n"
        "\\label{fig:lagsweep}\n\\end{figure}\n"),
}

POINTERS = [
    ("Allowed its \\SI{300}{\\second} lag, the committed estimate roughly\n"
     "halves the causal error, to \\num{24.2}/\\SI{11.5}{\\meter}. "
     "Fig.~\\ref{fig:errhist}\nshows the error history.",
     "Allowed its \\SI{300}{\\second} lag, the committed estimate roughly\n"
     "halves the causal error, to \\num{24.2}/\\SI{11.5}{\\meter}. "
     "Fig.~\\ref{fig:errhist}\nshows the error history, Fig.~\\ref{fig:track} "
     "the recovered track on the map,\nand Fig.~\\ref{fig:comp} the "
     "compensation the graph learned while recovering it."),
    ("nothing re-tuned. The committed estimate is the most accurate entry",
     "nothing re-tuned (Fig.~\\ref{fig:breadthbar}). The committed estimate "
     "is the most accurate entry"),
    ("sweeps it from \\SI{30}{} to \\SI{600}{\\second} on the four counted "
     "lines.",
     "sweeps it from \\SI{30}{} to \\SI{600}{\\second} on the four counted "
     "lines (Fig.~\\ref{fig:lagsweep})."),
]


def main():
    src = io.open(TEX, encoding="utf-8").read()
    src = re.sub(r"\\caption\{The cold-start problem.*?\}\n\\label\{fig:concept\}"
                 r"|\\caption\{Concept of the study.*?\}\n\\label\{fig:concept\}",
                 CONCEPT_CAP.replace("\\", "\\\\") + "\n\\\\label{fig:concept}",
                 src, count=1, flags=re.S)
    # insert figures after their anchors; do it in one pass per anchor so two
    # figures at the same anchor keep their order
    for lab in ("fig:comp", "fig:track", "fig:breadthbar", "fig:lagsweep"):
        anchor, block = FIGS[lab]
        if f"\\label{{{lab}}}" in src:
            continue
        i = src.index(anchor) + len(anchor)
        src = src[:i] + "\n\n" + block + src[i:]
    applied = 0
    for old, new in POINTERS:
        if old in src:
            src = src.replace(old, new)
            applied += 1
        else:
            print(f"  POINTER NOT FOUND: {' '.join(old.split())[:60]}")
    io.open(TEX, "w", encoding="utf-8").write(src)
    print(f"figures wired; pointers {applied}/{len(POINTERS)}")


if __name__ == "__main__":
    main()
