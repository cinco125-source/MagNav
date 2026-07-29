#!/bin/bash
# Control the cold-start diagnosis against the simplest competing explanation:
# the Tolles-Lawson prior is sigma = 1 per coefficient while the cabin field
# needs coefficients of several hundred, so the graph may simply be starting
# from a prior scaled for a compensated installation. Sweep that prior with a
# ZERO mean and no bootstrap. If a diffuse prior alone removes the divergence,
# the cold-start story is about prior scale, not minimum selection.
# Also runs --seed-only, which uses the bootstrap coefficients as the initial
# iterate while leaving the prior mean at zero, so the first window's data is
# not counted in the prior and in the measurement factors at once.
cd /mnt/c/Users/cin64/Desktop/MagNav
PY=~/gtsam_env/bin/python
G=research/gtsam_poc

for LINE in 1003_02 1003_08 1006_08 1007_02 1007_06; do
    D=~/magnav_data/line_${LINE}_full.h5
    for S in 10 100 1000; do
        TAG=_ps${S}_${LINE}_m4
        [ -f $G/gtsam_poc_result$TAG.txt ] && continue
        $PY -u $G/run_gtsam_decimated.py $D 300 4 10 $TAG --tl-sigma $S \
            > $G/x_ps${S}_${LINE}_m4.log 2>&1
        printf "%-9s sigma=%-5s " $LINE $S
        grep "warm=600" $G/gtsam_poc_result$TAG.txt
    done
    TAG=_so_${LINE}_m4
    if [ ! -f $G/gtsam_poc_result$TAG.txt ]; then
        $PY -u $G/run_gtsam_decimated.py $D 300 4 10 $TAG \
            --boot 300 --tl-init 300 --seed-only > $G/x_so_${LINE}_m4.log 2>&1
        printf "%-9s seed-only " $LINE
        grep "warm=600" $G/gtsam_poc_result$TAG.txt
    fi
done
echo PRIOR_SCALE_DONE
