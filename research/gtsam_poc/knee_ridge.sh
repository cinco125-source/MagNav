#!/bin/bash
# Same window sweep as knee_sweep.sh but started from the ridge-seeded
# compensation instead of zero. Comparing the two logs answers whether the
# accurate solution is also the cheaper one at every case and window length,
# which is what makes the failure detectable online from the objective alone.
cd /mnt/c/Users/cin64/Desktop/MagNav
PY=~/gtsam_env/bin/python
G=research/gtsam_poc
OUT=$G/knee_ridge_causal.log
: > $OUT

for LINE in 1003_02 1003_08 1006_08 1007_02 1007_06; do
    D=~/magnav_data/line_${LINE}_full.h5
    echo "########## $LINE ##########" >> $OUT
    for MAG in 4 5; do
        $PY -u $G/diag_window1.py $D $MAG 10 30,100,200,300,400,500,600,900 \
            --init-ridge >> $OUT 2>&1
    done
done
echo KNEE_RIDGE_CAUSAL_DONE >> $OUT
echo KNEE_RIDGE_CAUSAL_DONE
