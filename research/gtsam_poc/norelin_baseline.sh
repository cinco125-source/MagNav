#!/bin/bash
# The linear fixed-lag smoother control, on all eight counted cases, at both
# compensation priors.
#
# WHY. Section I concedes that for a linear-Gaussian model the fixed-lag MAP
# estimate coincides with classical fixed-lag smoothing, so smoothing per se is
# not the contribution. Every baseline in Table IV is nonetheless a causal
# filter, which leaves the smoothed column without an opponent at its own
# operating point and leaves V-A's mechanism claim -- that the estimators
# "differ only in that the filter commits each state once while the graph
# relinearizes and revises every state inside its lag" -- untested.
#
# --norelin is that opponent. Each new state is still linearized at Phi times
# the previous causal estimate, the point a filter linearizes at, and no state
# is relinearized afterwards. Same error model, same compensation states, same
# lag, same measurements, none of the revision. The gap to the full run is the
# formulation's share of the margin; the gap from there to the causal filters is
# what smoothing alone would have bought.
#
# WHAT THE SEGMENT ALREADY SHOWS (line 1007.06 Mag 5, 10 min, smoothed/realtime,
# INS 51.95; gtsam_poc_result_{rl,sb1_l}*.txt):
#
#                sigma_beta = 1            sigma_beta = 100
#   lag        full      norelin         full      norelin
#   300  15.11/19.96  28.12/38.65   7.88/15.49  8.31/15.28
#    30  16.10/18.81  33.74/38.65  14.31/16.19  13.45/15.28
#
# At sigma_beta = 100 relinearization is inert, which agrees with Table VIII's
# relinearize-every-variable row moving 1007.06 Mag 4 by a tenth of a metre:
# both ends of the knob do nothing. At sigma_beta = 1 the same switch costs a
# factor of 1.86 to 2.10, and norelin returns 38.65 m realtime at BOTH lags --
# ten times the look-ahead changes nothing, because what traps it is the
# linearization, not the amount of future data.
#
# So the reading to test at scale is that relinearization buys the basin rather
# than the metres, and that a wide compensation prior and relinearization are
# substitutes: widen the prior far enough and there is no bad branch left to
# escape. sigma_beta = 100 stays the reported operating point; the
# sigma_beta = 1 runs are mechanism evidence, not a proposed headline.
#
# READING THE OUTPUT. The clean comparison at sigma_beta = 1 is Mag 5. The full
# graph already diverges there on three Mag 4 cases (19457, 26811, 11262 m,
# gtsam_poc_result_dec300_*), which is why the paper widened the prior at all;
# where both columns diverge the pair says nothing. Expect the mechanism to show
# on the four sigma_beta = 1 Mag 5 cases and to be absent at sigma_beta = 100.
#
# COST. Sixteen runs, one per case per prior, about 40 s each on a 5240-state
# line. The full runs are reused: ps100_* and dec300_*.
set -e
cd /mnt/c/Users/cin64/Desktop/MagNav
PY=~/gtsam_env/bin/python
DATA=~/magnav_data
G=research/gtsam_poc
CASES="1007_06:4 1007_06:5 1007_02:4 1007_02:5 1003_02:4 1003_02:5 1003_08:4 1003_08:5"

for C in $CASES; do
    LINE=${C%:*}; MAG=${C#*:}
    for P in 100 1; do
        TAG=_nr${P}_${LINE}_m${MAG}
        [ -f $G/gtsam_poc_result$TAG.txt ] && { echo "=== $TAG cached ==="; continue; }
        EXTRA=""; [ "$P" = "100" ] && EXTRA="--tl-sigma 100"
        echo "=== $TAG ==="
        $PY -u $G/run_gtsam_decimated.py $DATA/line_${LINE}_full.h5 300 $MAG 10 $TAG \
            $EXTRA --norelin > $G/x${TAG}.log 2>&1 || { tail -20 $G/x${TAG}.log; exit 1; }
        grep "warm=600" $G/gtsam_poc_result$TAG.txt
    done
done

echo
echo "########## full vs norelin, smoothed / realtime [m] ##########"
val () {  # val <resultfile>
    [ -f "$1" ] || { printf '%13s' "-"; return; }
    printf '%13s' "$(grep -h 'warm=600' "$1" | sed 's/.*smoothed_drms=\([0-9.]*\) m  realtime_drms=\([0-9.]*\).*/\1\/\2/')"
}
printf '%-14s %13s %13s %13s %13s\n' case "sb1 full" "sb1 norelin" "sb100 full" "sb100 norelin"
for C in $CASES; do
    LINE=${C%:*}; MAG=${C#*:}
    printf '%-14s' "$LINE m$MAG"
    val $G/gtsam_poc_result_dec300_${LINE}_m${MAG}.txt
    val $G/gtsam_poc_result_nr1_${LINE}_m${MAG}.txt
    val $G/gtsam_poc_result_ps100_${LINE}_m${MAG}.txt
    val $G/gtsam_poc_result_nr100_${LINE}_m${MAG}.txt
    echo
done
echo
echo "Mag 5 at sigma_beta=1 is the clean comparison; ignore pairs where both diverge."
echo NORELIN_BASELINE_DONE
