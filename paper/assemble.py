#!/usr/bin/env python3
"""Assemble the restructured manuscript from the section drafts.

Replaces everything between the introduction's organization paragraph and the
appendices with, in order: draft_prelim (Problem Formulation + FGO),
draft_method (Methodology), draft_results_ab + tables + draft_results_cg
(Experiments), draft_conclusion. The observability material moves to a new
appendix built from the old Section IV's proposition and proof. The old
organization paragraph is rewritten to match.

The result is written to taes_incremental.tex, leaving taes_fgo_magnav.tex
untouched as the archive of the previous structure.

Run: python paper/assemble.py
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, "taes_fgo_magnav.tex")
NEW = os.path.join(HERE, "taes_incremental.tex")


def read(name):
    src = io.open(os.path.join(HERE, name), encoding="utf-8").read()
    # drop the staging header comments
    i = src.index("\\section")
    return src[i:]


ORG = r"""% P8: Organization
The remainder of this paper is organized as follows.
Section~\ref{sec:problem} states the problem in its classical batch form and
formalizes the cold start. Section~\ref{sec:fgo} sets out factor graph
optimization from first principles. Section~\ref{sec:method} develops the joint
system model, its factor graph, and the incremental fixed-lag solution.
Section~\ref{sec:results} describes the protocol and baselines and reports the
experiments. Section~\ref{sec:conclusion} concludes.
"""


def main():
    src = io.open(OLD, encoding="utf-8").read()

    # body: from the organization paragraph to the appendices
    i_org = src.index("% P8: Organization")
    i_app = src.index("\\appendices")

    prelim = read("draft_prelim.tex")
    method = read("draft_method.tex")
    results_ab = io.open(os.path.join(HERE, "draft_results_ab.tex"),
                         encoding="utf-8").read()
    results_ab = results_ab[results_ab.index("\\section"):]
    tables = io.open(os.path.join(HERE, "draft_tables.tex"),
                     encoding="utf-8").read()
    tables = tables[tables.index("%% ----"):]
    results_cg = io.open(os.path.join(HERE, "draft_results_cg.tex"),
                         encoding="utf-8").read()
    results_cg = results_cg[results_cg.index("\\subsection"):]
    conclusion = read("draft_conclusion.tex")

    # observability appendix from the old Section IV: keep the proposition,
    # remark and proof, reframed as supporting material.
    m = re.search(r"\\begin\{proposition\}.*?\\end\{proposition\}", src, re.S)
    prop = m.group(0) if m else ""
    m = re.search(r"\\begin\{remark\}\\label\{rem:floor\}.*?\\end\{remark\}",
                  src, re.S)
    remark = m.group(0) if m else ""
    m = re.search(r"\\section\{Proof of Proposition.*?(?=\\section|\\end\{document\})",
                  src, re.S)
    proof_app = m.group(0) if m else ""

    obs_app = (
        "\\section{Separability of Position and Compensation}\\label{app:obs}\n"
        "The linearized measurement~\\eqref{eq:measlin} carries a position error\n"
        "only through the map gradient and a compensation error only through the\n"
        "regressor, so over a window the two are distinguishable exactly when no\n"
        "position perturbation and coefficient perturbation produce the same\n"
        "residual sequence. The following condition makes that precise; it\n"
        "supports the discussion of Section~\\ref{sec:sysmodel}.\n\n"
        + prop + "\n\n" + remark + "\n")

    body = (ORG + "\n"
            + prelim.rstrip() + "\n\n"
            + method.rstrip() + "\n\n"
            + results_ab.rstrip() + "\n\n"
            + results_cg.rstrip() + "\n\n"
            + tables.rstrip() + "\n\n"
            + conclusion.rstrip() + "\n\n")

    out = src[:i_org] + body + src[i_app:]

    # insert the observability appendix before the Pinson appendix
    i_pin = out.index("\\section{Pinson Error-State Dynamics}")
    out = out[:i_pin] + obs_app + "\n" + out[i_pin:]

    io.open(NEW, "w", encoding="utf-8").write(out)
    print(f"wrote taes_incremental.tex ({out.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
