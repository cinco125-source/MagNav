#!/bin/bash
# The whole pipeline on the COMPENSATED stinger, mag_1_c.
#
# This is the setup a terrain-referenced navigation paper uses: a sensor whose
# measurement already matches the map, so navigation is the only problem. It is
# what park2024 works in, minus that paper's bias node.
#
# READ THIS BEFORE READING THE NUMBERS. On a clean sensor the graph and a linear
# fixed-lag smoother are expected to coincide, so column D decides whether there
# is anything here at all. The evidence: --norelin costs 5.5% at sigma_beta=100
# and a factor of 1.86 at sigma_beta=1, i.e. relinearization only pays when the
# problem is badly conditioned, and a clean sensor is the best-conditioned case
# there is -- it has no compensation nuisance whatsoever. The projected bound
# agrees: dropping the 19-column Tolles-Lawson subspace moves mu^-1/2 from 1.69
# to 1.45 m per nT of sigma, so the compensation was costing 1.2x and removing
# it buys almost nothing. If D comes back flat, the clean-sensor result is
# "a smoother beats a filter", which Section I already concedes is not a
# contribution, and this direction is closed.
#
# The comparison against the existing EKF on the same sensor (fgo_breadth.jl,
# warm=600, no TL states): 1003.02 21.5, 1003.08 17.7, 1006.08 22.3,
# 1007.02 25.6, 1007.06 20.0. Note the batch FGO figures in README section 2
# (14.3 / 14.0 on 1003.02) carry NO warm-up and cannot be set against these.
#
# FOUR COLUMNS PER LINE
#   A  clean sensor, TL states carried    -- our estimator unchanged
#   B  clean sensor, TL frozen at zero    -- pure navigation, no compensation
#   C  as A but --norelin                 -- the linear fixed-lag smoother
#   D  as B but --norelin                 -- the same, without compensation
# A vs B says whether the TL states are doing anything on a sensor that does not
# need them; if B is much worse than A the coefficients were absorbing map error
# rather than interference, which would also cast the cabin results in a
# different light. A vs C and B vs D are the contribution test.
set -e
cd /mnt/c/Users/cin64/Desktop/MagNav
PY=~/gtsam_env/bin/python
DATA=~/magnav_data
G=research/gtsam_poc
FLT() { case $1 in 1003*) echo Flt1003;; 1006*) echo Flt1006;; 1007*) echo Flt1007;; esac; }
MAP() { case $1 in 1003.02|1006.08|1007.02) echo Eastern_395;; 1003.08|1007.06) echo Renfrew_395;; esac; }
LINES="1003.02 1003.08 1007.02 1007.06"
# freezing the compensation: a negligible prior and a negligible walk pin every
# coefficient at zero, which is a no-compensation run without touching the code
FREEZE="--tl-sigma 1e-6 --tl-walk 1e-6"

for L in $LINES; do
    U=${L/./_}
    OUT=$DATA/line_${U}_full.h5
    if [ ! -f "$OUT" ] || ! $PY -c "import h5py,sys; sys.exit(0 if 'meas_mag1c' in h5py.File('$OUT','r') else 1)" 2>/dev/null; then
        echo "=== EXPORT $L (needs meas_mag1c) ==="
        $PY -u $G/export_full_line.py \
            $DATA/sgl_2020_train/$(FLT $L)_train.h5 \
            $DATA/ottawa_area_maps/$(MAP $L).h5 "$OUT" --line $L
    fi
done

run () {  # run <line_underscored> <tag> <extra...>
    local U=$1 TAG=$2; shift 2
    [ -f $G/gtsam_poc_result$TAG.txt ] && { echo "  $TAG cached"; return; }
    $PY -u $G/run_gtsam_decimated.py $DATA/line_${U}_full.h5 300 1c 10 $TAG \
        "$@" > $G/x$TAG.log 2>&1 || { tail -20 $G/x$TAG.log; exit 1; }
}

for L in $LINES; do
    U=${L/./_}
    echo "=== $L on mag_1_c ==="
    run $U _cl_A_$U --tl-sigma 100
    run $U _cl_B_$U $FREEZE
    run $U _cl_C_$U --tl-sigma 100 --norelin
    run $U _cl_D_$U $FREEZE --norelin
done

echo
echo "########## results, in the same form as every other table ##########"
python3 research/gtsam_poc/comparison_table.py
echo CLEAN_SENSOR_DONE
