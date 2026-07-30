#!/usr/bin/env python3
"""Table IV EKF-online column moves to the common sigma_beta=100 prior
(research/ekf_sig100_results.csv): bounded on all eight cases."""
import io

TEX = "taes_incremental.tex"
src = io.open(TEX, encoding="utf-8").read()
rows = [
 ("1007.06 & 4 & 46.7 & 48.8 & 42.7 & \\textbf{24.2} \\\\",
  "1007.06 & 4 & 46.3 & 48.8 & 42.7 & \\textbf{24.2} \\\\"),
 ("1007.06 & 5 & 17.8 & 18.3 & 16.0 & \\textbf{11.5} \\\\",
  "1007.06 & 5 & 17.6 & 18.3 & 16.0 & \\textbf{11.5} \\\\"),
 ("1007.02 & 4 & div. & 130.0 & 74.3 & \\textbf{28.4} \\\\",
  "1007.02 & 4 & 87.5 & 130.0 & 74.3 & \\textbf{28.4} \\\\"),
 ("1007.02 & 5 & 31.6 & 30.4 & 33.1 & \\textbf{15.8} \\\\",
  "1007.02 & 5 & 31.9 & 30.4 & 33.1 & \\textbf{15.8} \\\\"),
 ("1003.02 & 4 & div. & 101.0 & 58.2 & \\textbf{19.6} \\\\",
  "1003.02 & 4 & 155.9 & 101.0 & 58.2 & \\textbf{19.6} \\\\"),
 ("1003.02 & 5 & 28.1 & 29.5 & 21.9 & \\textbf{12.0} \\\\",
  "1003.02 & 5 & 28.7 & 29.5 & 21.9 & \\textbf{12.0} \\\\"),
 ("1003.08 & 4 & err. & 46.6 & 48.9 & \\textbf{25.0} \\\\",
  "1003.08 & 4 & 48.8 & 46.6 & 48.9 & \\textbf{25.0} \\\\"),
 ("1003.08 & 5 & 21.1 & 20.7 & 19.2 & \\textbf{10.5} \\\\",
  "1003.08 & 5 & 19.6 & 20.7 & 19.2 & \\textbf{10.5} \\\\"),
]
miss = 0
for o, n in rows:
    if o in src:
        src = src.replace(o, n)
    else:
        miss += 1
        print("MISS:", repr(o[:44]))
io.open(TEX, "w", encoding="utf-8").write(src)
print("done, misses:", miss)
