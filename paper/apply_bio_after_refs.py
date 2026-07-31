#!/usr/bin/env python3
"""Move the author biographies after the bibliography, so the back matter
reads appendices, acknowledgment, references, biographies."""
import io
import re

TEX = "taes_incremental.tex"
src = io.open(TEX, encoding="utf-8").read()

bios = re.findall(r"\\begin\{biography\}.*?\\end\{biography\}\n", src, re.S)
if len(bios) != 3:
    raise SystemExit(f"expected 3 biographies, found {len(bios)}")
for b in bios:
    src = src.replace(b, "")
block = "\n" + "\n".join(bios)

anchor = "\\bibliography{references}\n"
if anchor not in src:
    raise SystemExit("bibliography anchor not found")
src = src.replace(anchor, anchor + block, 1)
src = re.sub(r"\n{3,}", "\n\n", src)
io.open(TEX, "w", encoding="utf-8").write(src)
print("biographies moved after the bibliography")
