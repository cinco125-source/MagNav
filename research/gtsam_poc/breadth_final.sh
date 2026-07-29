#!/bin/bash
# Final incremental-smoother breadth sweep: 5 lines x Mag4/Mag5.
# lag = 300 s, 1 Hz decimated states, and a cold-start bootstrap consisting of
#   (a) a ridge Tolles-Lawson fit over the first lag-length of data, and
#   (b) one batch solve of that same first window,
# after which the smoother runs per-epoch. Both use only the data the smoother
# already holds, and the bootstrap window is one lag, so nothing is added to the
# committed latency. Ablations live in _dec300_* (nothing), _dec300tl_* (a only)
# and _dec300boot_* (b only, 900 s).
cd /mnt/c/Users/cin64/Desktop/MagNav
PY=~/gtsam_env/bin/python
G=research/gtsam_poc

for LINE in 1003_02 1003_08 1006_08 1007_02 1007_06; do
    D=~/magnav_data/line_${LINE}_full.h5
    for MAG in 4 5; do
        TAG=_dec300br_${LINE}_m$MAG
        if [ -f $G/gtsam_poc_result$TAG.txt ]; then
            echo "=== $LINE mag$MAG cached ==="; continue
        fi
        echo "=== $LINE mag$MAG (lag=300, boot=300+ridge) ==="
        $PY -u $G/run_gtsam_decimated.py $D 300 $MAG 10 $TAG --boot 300 --tl-init 300 \
            > $G/x_br_${LINE}_m$MAG.log 2>&1
        grep "warm=600" $G/gtsam_poc_result$TAG.txt
    done
done
echo BREADTH_FINAL_DONE
