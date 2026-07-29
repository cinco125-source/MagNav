#!/bin/bash
# Minimum-lag test for the Mag-4 cold start (line 1007.06).
# diag_window1.py locates the knee between 400 s and 500 s: the batch optimum of
# the first W states is 1109 m off at W=400 s and 57 m off at W=500 s, while
# Mag 5 is well posed at every W. If the divergence is the fixed-lag smoother
# committing states before the joint calibration+navigation problem is
# observable, then a lag past the knee should navigate and a lag before it
# should not.
cd /mnt/c/Users/cin64/Desktop/MagNav
PY=~/gtsam_env/bin/python
D=~/magnav_data/line_1007_06_full.h5
G=research/gtsam_poc

for LAG in 600 900 450; do
    $PY -u $G/run_gtsam_decimated.py $D $LAG 4 10 _dec${LAG}_1007_06_m4 \
        > $G/x_lag${LAG}_m4.log 2>&1
    echo "--- lag $LAG mag4 ---"
    grep "warm=600" $G/gtsam_poc_result_dec${LAG}_1007_06_m4.txt
done
echo LAG_KNEE_DONE
