#!/bin/bash
# Compensation as separate long-lived graph variables (run_gtsam_split.py).
#
# Stage 1 is a CONTROL, not a result: --beta-period 1 --no-beta-persist is the
# existing per-epoch scheme written in the split parameterization, so it must
# reproduce gtsam_poc_result_ps100_1007_06_m4.txt (24.19 smoothed / 42.72
# realtime at warm=600) to within the elimination-order difference. If it does
# not, stop -- the refactor is wrong and nothing below means anything.
#
# Stage 2 sweeps the compensation node period on the headline case and on the
# two cases where the smoother currently trails a causal filter
# (1007.02 Mag 5 at 33.14 vs EKF+TL+NN 30.4; 1003.08 Mag 4 at 48.85 vs 46.6).
# Stage 3 runs the winning period over all 8 counted cases.
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
echo "########## stage 2: compensation node period sweep ##########"
for T in 60 300 600 0; do
    for CASE in "1007_06 4" "1007_02 5" "1003_08 4"; do
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
