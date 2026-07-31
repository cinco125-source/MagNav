#!/usr/bin/env python3
"""Move the Proposition proof appendix up beside the other appendices, so the
back matter reads appendices, acknowledgment, references, biographies."""
import io
import re

TEX = "taes_incremental.tex"
src = io.open(TEX, encoding="utf-8").read()

m = re.search(r"\\section\{Proof of Proposition~\\ref\{prop:obs\}\}"
              r"\\label\{app:proof\}.*?\\hfill\$\\blacksquare\$\n", src, re.S)
if not m:
    raise SystemExit("proof appendix not found")
proof = m.group(0)
src = src[:m.start()] + src[m.end():]

anchor = "\n\\section*{Acknowledgment}"
if anchor not in src:
    raise SystemExit("acknowledgment anchor not found")
src = src.replace(anchor, "\n" + proof + anchor, 1)

# tidy the blank lines the removal left behind before the bibliography
src = re.sub(r"\n{3,}\\bibliographystyle", "\n\n\\\\bibliographystyle", src)
io.open(TEX, "w", encoding="utf-8").write(src)
print("proof appendix moved before the acknowledgment")
