#!/bin/bash
# Three questions the isotropic prior sweep leaves open.
#  1. sigma_S: the FOGM disturbance is fitted to 70 nT against a 3 nT prior in
#     the cold-start window solves, so it is the other under-scaled prior in the
#     same measurement equation. Does widening it alone help?
#  2. --tl-cols: the 19 regressor columns differ by four orders of magnitude in
#     scale, so one isotropic sigma cannot be right for all of them. Giving each
#     column a common prior on its CONTRIBUTION in nT is the dimensionally
#     consistent version. If that works at a physically sized setting, it is a
#     better recommendation than simply widening everything.
#  3. sigma_beta = 100 on Mag 5, to check the setting we recommend does not cost
#     accuracy on the channel that never needed it.
cd /mnt/c/Users/cin64/Desktop/MagNav
PY=~/gtsam_env/bin/python
G=research/gtsam_poc

for LINE in 1003_02 1003_08 1006_08 1007_02 1007_06; do
    D=~/magnav_data/line_${LINE}_full.h5
    for SS in 10 100; do
        TAG=_ss${SS}_${LINE}_m4
        [ -f $G/gtsam_poc_result$TAG.txt ] || {
            $PY -u $G/run_gtsam_decimated.py $D 300 4 10 $TAG --sigma-s $SS \
                > $G/x_ss${SS}_${LINE}_m4.log 2>&1
            printf "%-9s sigmaS=%-4s " $LINE $SS
            grep "warm=600" $G/gtsam_poc_result$TAG.txt; }
    done
    for SC in 100 1000; do
        TAG=_tc${SC}_${LINE}_m4
        [ -f $G/gtsam_poc_result$TAG.txt ] || {
            $PY -u $G/run_gtsam_decimated.py $D 300 4 10 $TAG --tl-cols $SC \
                > $G/x_tc${SC}_${LINE}_m4.log 2>&1
            printf "%-9s tlcols=%-4s " $LINE $SC
            grep "warm=600" $G/gtsam_poc_result$TAG.txt; }
    done
    TAG=_ps100_${LINE}_m5
    [ -f $G/gtsam_poc_result$TAG.txt ] || {
        $PY -u $G/run_gtsam_decimated.py $D 300 5 10 $TAG --tl-sigma 100 \
            > $G/x_ps100_${LINE}_m5.log 2>&1
        printf "%-9s m5 sigma=100 " $LINE
        grep "warm=600" $G/gtsam_poc_result$TAG.txt; }
done
echo PRIOR_STRUCTURE_DONE
