#!/bin/bash
# Robust-kernel probe for the Mag-4 divergence (line 1007.06, dec300).
# diag_resid.py shows the residual at the first linearization point (x = 0) is
# 1771 nT rms for Mag 4 against a Huber knee of 16.1 nT, so every measurement
# enters as an outlier before the Tolles-Lawson states can absorb the cabin
# field. The Julia reference applies its IRLS weights only from the second
# Gauss-Newton iteration and survives with and without the kernel.
cd /mnt/c/Users/cin64/Desktop/MagNav
PY=~/gtsam_env/bin/python
D=~/magnav_data/line_1007_06_full.h5
G=research/gtsam_poc

$PY -u $G/run_gtsam_decimated.py $D 300 4 10 _dec300norob_1007_06_m4 --norobust \
    > $G/x_norob_m4.log 2>&1
$PY -u $G/run_gtsam_decimated.py $D 300 4 10 _dec300hub10_1007_06_m4 --huber 10 \
    > $G/x_hub10_m4.log 2>&1
$PY -u $G/run_gtsam_decimated.py $D 300 4 10 _dec300hub100_1007_06_m4 --huber 100 \
    > $G/x_hub100_m4.log 2>&1
$PY -u $G/run_gtsam_decimated.py $D 300 5 10 _dec300norob_1007_06_m5 --norobust \
    > $G/x_norob_m5.log 2>&1
echo ROBUST_PROBE_DONE
