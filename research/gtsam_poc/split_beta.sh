#!/bin/bash
# Compensation as separate long-lived graph variables (run_gtsam_split.py).
#
# Stage 1 is a CONTROL, not a result: --beta-period 1 --no-beta-persist is the
# existing per-epoch scheme written in the split parameterization, so it should
# land on gtsam_poc_result_ps100_1007_06_m4.txt (24.19 smoothed / 42.72 realtime
# at warm=600).
#
# Expect close, not exact. The same control run against
# run_gtsam_decimated.py on the committed 1007.06 segment came in at -1.1% and
# -3.4% at a 300 s lag and -6.4% and -5.7% at 30 s, on the better side of the
# reference in all four columns: splitting a 37-dimensional variable into an 18
# and a 19 gives iSAM2 a finer relinearization decision. Take anything within
# about 7% either way. Stop and look at x_sp_ctrl_1007_06_m4.log only if it is
# consistently worse by more than 10%.
#
# Stage 2 sweeps the node period on the cold-start-degenerate cases: the two
# where the smoother trails a causal filter (1007.02 Mag 5 at 33.14 against the
# NN filter's 30.4; 1003.08 Mag 4 at 48.85 against 46.6) plus the two Mag 4
# lines where the causal margin is widest. Stage 3 takes the winner to all 8.
#
# Cost, measured on the 1007.06 segment: 7.4 ms/state at 2 compensation nodes
# against the reference's 9.0, rising to 40.3 at 20 nodes. At >=300 s periods a
# full 5240-state line is about 40 s, so stage 2 is minutes, not hours. The
# control at --beta-period 1 is the expensive one at roughly 150 ms/state.
#
# Everything runs at --tl-sigma 100, the prior the paper reports.
set -e
cd /mnt/c/Users/cin64/Desktop/MagNav
PY=~/gtsam_env/bin/python
DATA=~/magnav_data
G=research/gtsam_poc
run () {  # run <line_underscored> <mag> <tag> <extra args...>
    local LINE=$1 MAG=$2 TAG=$3; shift 3
    [ -f $G/gtsam_poc_result$TAG.txt ] && { echo "=== $TAG cached ==="; return; }
    echo "=== $TAG ==="
    $PY -u $G/run_gtsam_split.py $DATA/line_${LINE}_full.h5 300 $MAG 10 $TAG \
        --tl-sigma 100 "$@" > $G/x${TAG}.log 2>&1 || { tail -20 $G/x${TAG}.log; exit 1; }
    grep "warm=600" $G/gtsam_poc_result$TAG.txt
}

echo "########## stage 1: control (must match ps100) ##########"
run 1007_06 4 _sp_ctrl_1007_06_m4 --beta-period 1 --no-beta-persist
echo "reference ps100 1007_06 m4:"
grep "warm=600" $G/gtsam_poc_result_ps100_1007_06_m4.txt

echo
echo "########## stage 2: node period on the HARD cases ##########"
# The 1007.06 segment sweep (gtsam_poc_result_rl300_t*.txt) found no trend on
# Mag 5: 30/60/150/300/0 s gave 9.98/7.64/8.75/7.36/7.80 smoothed against a
# 7.79 control, with the scatter from node placement wider than any gap to the
# reference. That is the easy channel, where the reference already corrects
# 51.95 m of drift down to 15.49. The hypothesis is about the degenerate
# cold start, so spend the runs on Mag 4 and on the two cases that trail a
# causal filter, and skip the short periods, which cost 4x and showed nothing.
for T in 300 600 0; do
    for CASE in "1007_06 4" "1007_02 5" "1003_08 4" "1003_02 4"; do
        set -- $CASE
        run $1 $2 _sp_t${T}_$1_m$2 --beta-period $T
    done
done

echo
echo "########## stage 3: breadth at the chosen period ##########"
# edit T_BEST after reading stage 2
T_BEST=${T_BEST:-600}
for CASE in "1007_06 4" "1007_06 5" "1007_02 4" "1007_02 5" \
            "1003_02 4" "1003_02 5" "1003_08 4" "1003_08 5"; do
    set -- $CASE
    run $1 $2 _sp_best_$1_m$2 --beta-period $T_BEST
done

echo
echo "########## summary ##########"
grep -H "warm=600" $G/gtsam_poc_result_sp_*.txt
echo SPLIT_BETA_DONE
