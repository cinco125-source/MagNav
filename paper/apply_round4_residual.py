#!/usr/bin/env python3
"""Residual round-4 fixes the main script missed (line-broken strings)."""
import io

TEX = "paper/taes_incremental.tex"
src = io.open(TEX, encoding="utf-8").read()

pairs = [
    ("\\SIrange{6.0}{9.0}{\\milli\\second} per state at a\n"
     "        \\SI{300}{\\second} lag",
     "\\SIrange{7.3}{9.0}{\\milli\\second} per state at a\n"
     "        \\SI{300}{\\second} lag"),
    ("\\SIrange{1729}{2081}{\\nano\\tesla} root-mean-square over the first",
     "\\SIrange{1729}{1978}{\\nano\\tesla} root-mean-square over the first"),
    ("costs \\SIrange{6.0}{9.0}{\\milli\\second} per state in an unoptimized",
     "costs \\SIrange{7.3}{9.0}{\\milli\\second} per state in an unoptimized"),
]
for old, new in pairs:
    assert old in src, repr(old[:60])
    src = src.replace(old, new)
io.open(TEX, "w", encoding="utf-8").write(src)
print("3 residual fixes applied")

for bad in ["6.0}{9.0", "2081", "strongest published", "single design variable",
            "ten line-magnetometer", "five survey", "solve it on board",
            "MPF{+}TL (best", "committed"]:
    if bad in src:
        print("LEFTOVER:", bad)
print("leftover scan done")
