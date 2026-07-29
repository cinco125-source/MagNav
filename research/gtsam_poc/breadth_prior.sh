#!/bin/bash
# Breadth sweep with the compensation prior scaled to the installation.
# The nominal TL prior is sigma = 1 per coefficient, while an uncompensated
# cabin magnetometer needs coefficients of 550 to 720. Widening that prior with
# a ZERO mean and no bootstrap removes the cold-start divergence on every Mag 4
# line (prior_scale.log). This runs the full 5 lines x Mag4/Mag5 at sigma = 1000,
# i.e. an effectively uninformative compensation prior, which is the
# configuration the paper should report.
cd /mnt/c/Users/cin64/Desktop/MagNav
PY=~/gtsam_env/bin/python
G=research/gtsam_poc

for LINE in 1003_02 1003_08 1006_08 1007_02 1007_06; do
    D=~/magnav_data/line_${LINE}_full.h5
    for MAG in 4 5; do
        TAG=_pr1k_${LINE}_m$MAG
        [ -f $G/gtsam_poc_result$TAG.txt ] && { echo "=== $LINE m$MAG cached ==="; continue; }
        $PY -u $G/run_gtsam_decimated.py $D 300 $MAG 10 $TAG --tl-sigma 1000 \
            > $G/x_pr1k_${LINE}_m$MAG.log 2>&1
        printf "%-9s m%s " $LINE $MAG
        grep "warm=600" $G/gtsam_poc_result$TAG.txt
    done
done
echo BREADTH_PRIOR_DONE
