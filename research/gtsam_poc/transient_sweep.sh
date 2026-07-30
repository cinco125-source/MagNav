#!/bin/bash
# Review round 4: (1) first-600 s transient metrics of the recommended config
# on every counted case, (2) initial-position prior sweep on 1007.06.
# Recommended config = lag 300, K 10, --tl-sigma 100 (prior_structure.sh).
set -u
G=research/gtsam_poc
PY=$HOME/gtsam_env/bin/python
DATA=$HOME/magnav_data

# (1) transient metrics, all counted cases (tag _tr_ to avoid clobbering ps100)
for LINE in 1007_06 1007_02 1003_02 1003_08; do
    D=$DATA/line_${LINE}_full.h5
    for M in 4 5; do
        TAG=_tr_${LINE}_m${M}
        [ -f $G/gtsam_poc_result$TAG.txt ] || {
            $PY -u $G/run_gtsam_decimated.py $D 300 $M 10 $TAG --tl-sigma 100 \
                > $G/x_tr_${LINE}_m${M}.log 2>&1
            printf "%-9s m%s " $LINE $M
            grep "transient" $G/gtsam_poc_result$TAG.txt; }
    done
done

# (2) initial-position prior sweep, 1007.06 both mags
D=$DATA/line_1007_06_full.h5
for PS in 10 50 100; do
    for M in 4 5; do
        TAG=_pos${PS}_1007_06_m${M}
        [ -f $G/gtsam_poc_result$TAG.txt ] || {
            $PY -u $G/run_gtsam_decimated.py $D 300 $M 10 $TAG \
                --tl-sigma 100 --pos-sigma $PS \
                > $G/x_pos${PS}_1007_06_m${M}.log 2>&1
            printf "pos=%-4s m%s " $PS $M
            grep -h "warm=600\|transient" $G/gtsam_poc_result$TAG.txt; }
    done
done
echo TRANSIENT_SWEEP_DONE
