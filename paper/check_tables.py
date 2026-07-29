#!/usr/bin/env python3
"""Check that every tabular row supplies the number of cells its preamble declares.

LaTeX is not installed in the analysis environment, so this catches the most
common breakage introduced when editing tables by hand.

Usage: check_tables.py [file.tex]
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def brace_span(s, i):
    """Content of the brace group starting at s[i], and the index past its close."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError(f"unbalanced brace at {i}")


def cells(row):
    return len(re.findall(r"(?<!\\)&", row)) + 1


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "taes_fgo_magnav.tex")
    src = open(path, encoding="utf-8").read()
    bad = 0
    for m in re.finditer(r"\\begin\{tabular\}(?:\[[^\]]*\])?\{", src):
        preamble, after = brace_span(src, m.end() - 1)
        end = src.index("\\end{tabular}", after)
        line0 = src[:m.start()].count("\n") + 1
        pre = re.sub(r"@\{[^}]*\}", "", preamble)
        pre = re.sub(r"p\{[^}]*\}", "p", pre)
        ncol = sum(1 for c in pre if c in "lcrp")
        body = re.sub(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{[^}]*\}",
                      lambda x: "&" * (int(x.group(1)) - 1), src[after:end])
        for row in body.split("\\\\"):
            row = re.sub(r"\\(?:toprule|midrule|bottomrule)", "", row)
            row = re.sub(r"\\cmidrule(?:\([^)]*\))?\{[^}]*\}", "", row)
            row = row.strip()
            if not row or row.startswith("%"):
                continue
            n = cells(row)
            if n != ncol:
                bad += 1
                flat = " ".join(row.split())
                print(f"line {line0}: {n} cells vs {ncol} columns :: {flat[:80]}")
    print(f"tabular issues: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
