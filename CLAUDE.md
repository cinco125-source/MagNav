# MagNav — working rules for this repository

This repo carries a research extension to MagNav.jl (factor-graph aeromagnetic
compensation + navigation) and a TAES manuscript built on it. Read
`research/AUDIT_RESTART.md` before touching `src/fgo*.jl`, `research/`, or
`paper/` — it lists the open defects and the verification ladder.

## Execution environment

Work in WSL (gtsam has no Windows wheels; the flight data lives there).

| What | How |
|---|---|
| Julia package + tests | `julia --project=. -e 'using Pkg; Pkg.test()'` |
| One test file | `julia --project=. -e 'using TestItemRunner; @run_package_tests filter=ti->occursin("fgo",ti.name)'` |
| Quick REPL probe | `julia --project=. -e 'using MagNav; ...'` |
| Julia research script | `julia --project=. research/<name>.jl` |
| GTSAM (paper engine) | `~/gtsam_env/bin/python research/gtsam_poc/run_gtsam_decimated.py <line.h5> <lag_s> <mag> <dec> <tag>` |
| Line export (no Julia) | `python research/gtsam_poc/export_full_line.py <Flt*_train.h5> <map.h5> <out.h5>` |
| Raw data | `~/magnav_data/` — SGL 2020 flights + Ottawa maps (Julia lazy artifacts fetch the rest) |

## Rules

1. **No number without a run.** Every figure that reaches a `.md`, `.tex`,
   commit message, or chat reply must come from a script in this repo that was
   executed in this session. Never copy a number from memory, from an earlier
   conversation, or from `research/README.md` — those are claims, not evidence.
   If a number cannot be reproduced right now, write "unverified" next to it.

2. **Run it, then say it.** After editing code, execute the affected test or
   script locally and paste the real output. "This should work" is not a result.

3. **Name the implementation.** Two independent implementations exist and they
   are not equivalent: Julia `src/fgo.jl` / `src/fgo_online.jl` / `src/fgo_sensor.jl`
   (CI-tested, part of the package) and Python `research/gtsam_poc/*.py`
   (GTSAM/iSAM2, produced the manuscript's headline numbers). State which one any
   number came from, every time.

4. **CI is a check, not the development loop.** Twenty-seven workflows exist
   because CI was previously used as a REPL, at roughly one hour per iteration.
   Run locally first; push only what already passes.

5. **Say "I don't know."** When a claim cannot be verified from the code or a
   run, say so and state what would settle it. Do not fill the gap by inference.

6. **The manuscript follows the code, never the reverse.** If a run contradicts
   a sentence in `paper/`, the sentence changes.

## Conversation

Korean for discussion, English for code / math / LaTeX. Honest assessment over
encouragement — say when a result is weak, an approach is incremental, or a
claim will not survive review.
