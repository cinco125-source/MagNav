#!/bin/bash
# Ablation around the PROPOSED configuration, not around the broken one.
# Reference: line 1007.06, both cabin magnetometers, per-epoch incremental
# smoother, lag 300 s, states at 1 Hz (K = 10), Huber kernel, sigma_beta = 100.
# Each run removes or changes exactly one element of that configuration.
cd /mnt/c/Users/cin64/Desktop/MagNav
PY=~/gtsam_env/bin/python
G=research/gtsam_poc
D=~/magnav_data/line_1007_06_full.h5
# Every run states its own full argument list. The option parser reads the FIRST
# occurrence of a flag, so a shared prefix would silently win over a per-variant
# override and the prior rows would all repeat the reference.
BASE=""

run () {   # run <tag> <extra args...>
    TAG=$1; shift
    for MAG in 4 5; do
        T=_ab${TAG}_m$MAG
        [ -f $G/gtsam_poc_result$T.txt ] && continue
        $PY -u $G/run_gtsam_decimated.py $D 300 $MAG 10 $T $BASE "$@" \
            > $G/x_ab${TAG}_m$MAG.log 2>&1
    done
    printf "%-14s " $TAG
    for MAG in 4 5; do
        grep -h "warm=600" $G/gtsam_poc_result_ab${TAG}_m$MAG.txt \
            | sed -E "s/.*smoothed_drms=([0-9.]+) m  realtime_drms=([0-9.]+).*/m$MAG \1 (\2)/" \
            | tr "\n" " "
    done
    echo
}

# reference configuration
run ref        --tl-sigma 100

# compensation prior
run prior1     --tl-sigma 1
run prior10    --tl-sigma 10
run prior1k    --tl-sigma 1000
run priorcol   --tl-cols 100

# compensation random walk
run walk01     --tl-sigma 100 --tl-walk 0.1
run walk10     --tl-sigma 100 --tl-walk 10

# FOGM disturbance prior
run fogm10     --tl-sigma 100 --sigma-s 10

# robust kernel
run norobust   --tl-sigma 100 --norobust
run huber10    --tl-sigma 100 --huber 10

# measurement noise and process-noise floor
run rsig65     --tl-sigma 100 --rsigma 65
run qf16       --tl-sigma 100 --qfloor 1e-16

# relinearization and measurement density
run fullrelin  --tl-sigma 100 --full-relin
run dense      --tl-sigma 100 --dense-meas

echo ABLATION_DONE
