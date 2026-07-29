#!/bin/bash
# Sensitivity of the cold-start bootstrap to its length. The nominal setting is
# one lag (300 s); shorter bootstraps sit inside the region where the window
# optimum is far from truth (Table knee), longer ones cost startup latency.
cd /mnt/c/Users/cin64/Desktop/MagNav
PY=~/gtsam_env/bin/python
G=research/gtsam_poc

for LINE in 1007_06 1003_02; do
    D=~/magnav_data/line_${LINE}_full.h5
    for L in 100 150 300 600; do
        TAG=_bl${L}_${LINE}_m4
        $PY -u $G/run_gtsam_decimated.py $D 300 4 10 $TAG --boot $L --tl-init $L \
            > $G/x_bl${L}_${LINE}_m4.log 2>&1
        printf "%s boot=%s  " $LINE $L
        grep "warm=600" $G/gtsam_poc_result$TAG.txt
    done
done
echo BOOT_LEN_DONE
