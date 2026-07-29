#!/bin/bash
# Controls for the two remaining alternative explanations of the cold-start
# divergence, both run at the NOMINAL compensation prior (sigma_beta = 1) so
# that only the candidate under test changes.
#   --rsigma: the post-fit residual on the cabin magnetometers is 47 to 153 nT
#             against an assumed 12 nT, so the measurement noise is
#             mis-specified. If that is what drives the divergence, matching R
#             to the residual should remove it.
#   --qfloor: the 1e-20 conditioning floor added to the process noise is the
#             other quantity that could be doing accidental work.
cd /mnt/c/Users/cin64/Desktop/MagNav
PY=~/gtsam_env/bin/python
G=research/gtsam_poc

for LINE in 1003_02 1003_08 1006_08 1007_02 1007_06; do
    D=~/magnav_data/line_${LINE}_full.h5
    for RS in 65 150; do
        TAG=_rs${RS}_${LINE}_m4
        [ -f $G/gtsam_poc_result$TAG.txt ] && continue
        $PY -u $G/run_gtsam_decimated.py $D 300 4 10 $TAG --rsigma $RS \
            > $G/x_rs${RS}_${LINE}_m4.log 2>&1
        printf "%-9s rsigma=%-4s " $LINE $RS
        grep "warm=600" $G/gtsam_poc_result$TAG.txt
    done
done

for QF in 1e-16 1e-24; do
    TAG=_qf${QF}_1007_06_m4
    [ -f $G/gtsam_poc_result$TAG.txt ] && continue
    $PY -u $G/run_gtsam_decimated.py ~/magnav_data/line_1007_06_full.h5 300 4 10 \
        $TAG --qfloor $QF > $G/x_qf${QF}_1007_06_m4.log 2>&1
    printf "1007_06   qfloor=%-6s " $QF
    grep "warm=600" $G/gtsam_poc_result$TAG.txt
done
echo ALT_CAUSES_DONE
