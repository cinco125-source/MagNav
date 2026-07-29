#!/bin/bash
# Cold-start observability knee for every line x magnetometer.
# For each case: (a) the uncompensated residual scale the calibration has to
# absorb (diag_resid.py), (b) the position error of the batch optimum of the
# first W states (diag_window1.py). The smallest W whose batch optimum is close
# to truth is the minimum usable fixed-lag horizon for that case.
cd /mnt/c/Users/cin64/Desktop/MagNav
PY=~/gtsam_env/bin/python
G=research/gtsam_poc
OUT=$G/knee_sweep.log
: > $OUT

for LINE in 1003_02 1003_08 1006_08 1007_02 1007_06; do
    D=~/magnav_data/line_${LINE}_full.h5
    echo "########## $LINE ##########" >> $OUT
    $PY -u $G/diag_resid.py $D >> $OUT 2>&1
    for MAG in 4 5; do
        $PY -u $G/diag_window1.py $D $MAG 10 30,100,200,300,400,500,600,900 \
            >> $OUT 2>&1
    done
done
echo KNEE_SWEEP_DONE >> $OUT
echo KNEE_SWEEP_DONE
