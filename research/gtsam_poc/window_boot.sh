#!/bin/bash
# Does the cold-start bootstrap help the SLIDING WINDOW too, in the same code
# base? The first window of the window smoother is itself a 300 s batch from a
# zero compensation, so the same spurious minimum should hit it, and the same
# ridge seed should fix it. Two hard cabin-magnetometer cases, with and without.
cd /mnt/c/Users/cin64/Desktop/MagNav
PY=~/gtsam_env/bin/python
G=research/gtsam_poc

for LINE in 1007_06 1003_02; do
    D=~/magnav_data/line_${LINE}_full.h5
    $PY -u $G/run_gtsam_window.py $D _win_${LINE}_m4 4 \
        > $G/x_win_${LINE}_m4.log 2>&1
    echo "--- window $LINE mag4 cold ---"
    tail -5 $G/gtsam_poc_result_window_win_${LINE}_m4.txt
    $PY -u $G/run_gtsam_window.py $D _winboot_${LINE}_m4 4 --tl-init 300 \
        > $G/x_winboot_${LINE}_m4.log 2>&1
    echo "--- window $LINE mag4 bootstrapped ---"
    tail -5 $G/gtsam_poc_result_window_winboot_${LINE}_m4.txt
done
echo WINDOW_BOOT_DONE
