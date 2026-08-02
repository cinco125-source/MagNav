#!/bin/bash
# Re-score the baselines over the WHOLE line, transient included.
#
# WHY. Every DRMS in the tables excludes the first 600 s. That is applied to
# every column alike, so the comparison is internally fair, but it sits oddly
# with a paper whose claim is about cold start: the transient is part of what a
# cold-start estimator has to deliver, not an exemption. The manuscript does
# report the window -- Table tab:transient gives DRMS and peak inside it, and
# says plainly that the causal output is transiently worse than coasting -- but
# that table carries only the proposed estimator. Nobody has measured how the
# online-TL EKF or the NN filter fare in the same window, so nobody knows which
# way a whole-line comparison would go.
#
# It is not obvious that including it hurts us. On the proposed estimator the
# transient costs the smoothed output almost nothing (1003.02 Mag 5: 12.04 m
# post-warm-up against 12.02 m whole-line) and the causal output 17 to 43%. The
# baselines start equally cold, so they pay too; the question is how much.
#
# WHAT RUNS. Only the warm=0 pass, since the warm=600 CSVs are already
# committed. Outputs go to *_warm0.csv and never touch the existing files.
# Roughly 20 minutes: the estimators are re-run because their DRMS helpers
# return a scalar and are called from ten other scripts, so threading a second
# cut-off through them was the riskier change.
#
# A cheap internal check afterwards: for any case, combining the manuscript's
# in-window DRMS with the post-warm-up DRMS by duration must reproduce the
# whole-line value. On 1003.02 Mag 5 causal that is
# sqrt((10*40.0^2 + 53*21.86^2)/63) = 25.6 against a measured 25.60.
set -e
cd "$(dirname "$0")/.."
JULIA=${JULIA:-julia}

echo "########## breadth baselines, whole line ##########"
DRMS_WARM=0 $JULIA --project=. research/fgo_breadth.jl

echo
echo "########## online-TL EKF at the shared prior, whole line ##########"
DRMS_WARM=0 $JULIA --project=. research/ekf_sig100.jl

echo
echo "########## both conventions side by side ##########"
python3 research/gtsam_poc/comparison_table.py
echo FULL_LINE_BASELINES_DONE
