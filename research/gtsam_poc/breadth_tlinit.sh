#!/bin/bash
# Incremental fixed-lag breadth sweep WITH the Tolles-Lawson bootstrap.
# 5 lines x Mag4/Mag5, lag = 300 s, 1 Hz decimated states, DRMS after 600 s.
# The paired no-bootstrap runs (tag _dec300_*) are the ablation.
cd /mnt/c/Users/cin64/Desktop/MagNav
PY=~/gtsam_env/bin/python
G=research/gtsam_poc

for LINE in 1003_02 1003_08 1006_08 1007_02 1007_06; do
    D=~/magnav_data/line_${LINE}_full.h5
    for MAG in 4 5; do
        TAG=_dec300tl_${LINE}_m$MAG
        if [ -f $G/gtsam_poc_result$TAG.txt ]; then
            echo "=== $LINE mag$MAG cached ==="; continue
        fi
        echo "=== $LINE mag$MAG (lag=300, TL bootstrap 300 s) ==="
        $PY -u $G/run_gtsam_decimated.py $D 300 $MAG 10 $TAG --tl-init 300 \
            > $G/x_tl_${LINE}_m$MAG.log 2>&1
        grep "warm=600" $G/gtsam_poc_result$TAG.txt
    done
done
echo BREADTH_TLINIT_DONE
