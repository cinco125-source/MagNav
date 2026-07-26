#!/bin/bash
# Per-epoch incremental (ISAM2, lag=30 s) breadth sweep: 5 lines x Mag4/Mag5,
# mirroring research/fgo_breadth.jl's protocol (DRMS after 600 s warm-up).
# Exports each line once (both mags in the h5), then runs the smoother per mag.
set -e
cd /mnt/c/Users/cin64/Desktop/MagNav
PY=~/gtsam_env/bin/python
DATA=~/magnav_data
FLT() { case $1 in 1003*) echo Flt1003;; 1006*) echo Flt1006;; 1007*) echo Flt1007;; esac; }
MAP() { case $1 in 1003.02|1006.08|1007.02) echo Eastern_395;; 1003.08|1007.06) echo Renfrew_395;; esac; }

for LINE in 1003.02 1003.08 1006.08 1007.02 1007.06; do
    OUT=$DATA/line_${LINE/./_}_full.h5
    if [ ! -f "$OUT" ] || ! $PY -c "import h5py,sys; sys.exit(0 if 'meas_mag4' in h5py.File('$OUT','r') else 1)" 2>/dev/null; then
        echo "=== EXPORT $LINE ($(FLT $LINE), $(MAP $LINE)) ==="
        $PY -u research/gtsam_poc/export_full_line.py \
            $DATA/sgl_2020_train/$(FLT $LINE)_train.h5 \
            $DATA/ottawa_area_maps/$(MAP $LINE).h5 \
            "$OUT" --line $LINE
    else
        echo "=== EXPORT $LINE cached ==="
    fi
done

for LINE in 1003.02 1003.08 1006.08 1007.02 1007.06; do
    for MAG in 4 5; do
        TAG=_lag30_${LINE/./}_m$MAG
        if [ -f research/gtsam_poc/gtsam_poc_result$TAG.txt ]; then
            echo "=== RUN $LINE mag$MAG cached ==="; continue
        fi
        echo "=== RUN $LINE mag$MAG (lag=30) ==="
        $PY -u research/gtsam_poc/run_gtsam.py $DATA/line_${LINE/./_}_full.h5 30 $MAG $TAG
    done
done
echo ALL_DONE
grep -H "warm=600" research/gtsam_poc/gtsam_poc_result_lag30_*.txt
