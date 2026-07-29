#!/usr/bin/env python3
"""Static sanity check for the manuscript: environment/brace/label balance."""
import re, sys

src = open(sys.argv[1], encoding="utf-8").read()

envs = {}
for m in re.finditer(r"\\(begin|end)\{(\w+\*?)\}", src):
    kind, env = m.group(1), m.group(2)
    envs.setdefault(env, 0)
    envs[env] += 1 if kind == "begin" else -1
bad = {k: v for k, v in envs.items() if v != 0}
print("unbalanced envs:", bad if bad else "none")

total = 0
for line in src.splitlines():
    line = re.sub(r"(?<!\\)%.*", "", line)
    line = line.replace(r"\{", "").replace(r"\}", "")
    total += line.count("{") - line.count("}")
print("net brace balance:", total)

print("documentclass:", src.count(r"\documentclass"),
      " end{document}:", src.count(r"\end{document}"))

labels = re.findall(r"\\label\{([^}]+)\}", src)
refs = set(re.findall(r"\\(?:ref|eqref|autoref|Cref|cref)\{([^}]+)\}", src))
dup = [l for l in set(labels) if labels.count(l) > 1]
missing = sorted(r for rr in refs for r in rr.split(",") if r not in set(labels))
print("dup labels:", dup if dup else "none")
print("refs w/o label:", missing if missing else "none")

# A non-raw Python replacement string once turned \beta into a literal backspace
# here, which pdflatex rejects outright and which no other check notices.
ctrl = [(i, ord(c)) for i, c in enumerate(src)
        if ord(c) < 32 and c not in "\n\t"]
if ctrl:
    for i, o in ctrl[:5]:
        print(f"control char U+{o:04X} on line {src[:i].count(chr(10)) + 1}: "
              f"{src[max(0, i - 40):i + 15]!r}")
print("control chars:", len(ctrl) if ctrl else "none")
nonascii = sorted({c for c in src if ord(c) > 126})
print("non-ascii chars:", nonascii if nonascii else "none")

cites = set()
for m in re.finditer(r"\\cite[tp]?\{([^}]+)\}", src):
    cites.update(k.strip() for k in m.group(1).split(","))
try:
    bib = open(sys.argv[2], encoding="utf-8").read()
    bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
    miss = sorted(cites - bibkeys)
    print("cites w/o bib entry:", miss if miss else "none")
except IndexError:
    pass
