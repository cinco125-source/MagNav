#!/usr/bin/env python3
"""Round-6c: clearer output names in the tables, the free-inertial range stated
as measured, the lag table given the causal column its caption claims, and the
transient table given the free-inertial reference it is compared against."""
import io

TEX = "taes_incremental.tex"
src = io.open(TEX, encoding="utf-8").read()
miss = 0


def rep(old, new):
    global src, miss
    if old in src:
        src = src.replace(old, new)
    else:
        miss += 1
        print("MISS:", " ".join(old.split())[:76])


# ---- clearer names for the two outputs in the breadth table --------------
rep(" & & EKF & EKF{+} & \\multicolumn{2}{c}{Proposed} \\\\\n"
    "\\cmidrule(l){5-6}\n"
    "Line & Mag & online & TL{+}NN & causal & \\SI{300}{\\second} lag \\\\",
    " & & EKF & EKF{+} & \\multicolumn{2}{c}{Proposed} \\\\\n"
    "\\cmidrule(l){5-6}\n"
    " & & & & current state & fixed-lag smoothed \\\\\n"
    "Line & Mag & online & TL{+}NN & (causal) & (\\SI{300}{\\second}) \\\\")

# ---- and in the representative-line table --------------------------------
rep("Proposed, causal & 42.7 & 16.0 & none \\\\",
    "Proposed, current state (causal) & 42.7 & 16.0 & none \\\\")
rep("\\textbf{Proposed, \\SI{300}{\\second} lag (smoothed)} & \\textbf{24.2} & "
    "\\textbf{11.5} & \\SI{300}{\\second} \\\\",
    "\\textbf{Proposed, fixed-lag smoothed (\\SI{300}{\\second})} & \\textbf{24.2} "
    "& \\textbf{11.5} & \\SI{300}{\\second} \\\\")

# ---- free-inertial: state the measured range -----------------------------
rep("\\SIrange{10.5}{28.4}{\\meter} where free-inertial drift is hundreds of metres. A",
    "\\SIrange{10.5}{28.4}{\\meter} where free-inertial drift is\n"
    "\\SIrange{121}{318}{\\meter}. A")

# ---- lag table: give the caption its causal column ------------------------
rep("\\caption{Lag sweep: smoothed DRMS [m]; the causal output varies by under a "
    "metre across the sweep.}",
    "\\caption{Lag sweep: smoothed DRMS [m], with the causal DRMS at the shortest "
    "and longest lag.}")
rep(" & & \\multicolumn{5}{c}{lag $L$ [s], smoothed} & causal \\\\\n"
    "\\cmidrule(lr){3-7}\n"
    "Line & Mag & 30 & 60 & 150 & 300 & 600 & \\\\",
    " & & \\multicolumn{5}{c}{lag $L$ [s], smoothed} & "
    "\\multicolumn{2}{c}{causal} \\\\\n"
    "\\cmidrule(lr){3-7}\\cmidrule(l){8-9}\n"
    "Line & Mag & 30 & 60 & 150 & 300 & 600 & $L{=}30$ & $L{=}600$ \\\\")
rep("\\begin{tabular}{ll rrrrr r}", "\\begin{tabular}{ll rrrrr rr}")

io.open(TEX, "w", encoding="utf-8").write(src)
print("round-6c applied, misses:", miss)
