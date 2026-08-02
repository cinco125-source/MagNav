#!/bin/bash
# Achieved error over the information bound, on all eight counted cases.
#
# WHY THIS FIRST. On the committed 1007.06 segment (Mag 5) the smoother reaches
# 7.88 m against a bound of 7.42 m: 1.06x, and 1.01x once the disturbance is
# retuned to what the residual actually shows. That single ratio explains why
# every knob tried this session moved the case by a few percent -- the noise
# model 4.8%, relinearization inert at sigma_beta = 100, the compensation node
# period no trend at all. There were only six percent to move.
#
# The ratio is therefore the triage number, and it is worth having on all eight
# cases before any more modelling effort is spent. Near 1.0 the case is
# information-limited and nothing in the estimator will help it. Well above 1.0
# is where the remaining work is, and the prediction to test is that the Mag 5
# cases sit near 1.0 while the Mag 4 cold starts do not: Mag 4 carries nine
# times the interference (1771 nT rms against 197) and diverges outright at the
# tight prior, which is the same place relinearization was worth a factor of two
# (gtsam_poc_result_sb1_l*). If that holds, the remaining work is one regime,
# not eight cases.
#
# WHAT IT DOES. For each case: make sure the sigma_beta = 100 estimate exists,
# re-running it if the npz was cleaned, then measure the unexplained field, fit
# the disturbance correlation time, evaluate mu^-1/2 at the assumed and the
# measured sigma, and report achieved over bound. The audit itself runs no
# estimator and takes seconds; a missing npz costs about 40 s to regenerate.
#
# Pairs with norelin_baseline.sh, which asks a different question about the same
# eight cases: not how much information is available, but how much of the margin
# comes from relinearizing rather than from smoothing. Run both; they share the
# ps100 runs and neither invalidates the other.
# THE RATIO DISCRIMINATES. Four configurations on the same segment against the
# same 7.42 m bound, from npz already in the repository:
#
#   sigma_beta = 100, relinearizing   7.88 m   1.06
#   sigma_beta = 100, --norelin       8.31 m   1.12
#   sigma_beta =   1, relinearizing  15.11 m   2.04
#   sigma_beta =   1, --norelin      28.12 m   3.79
#
# The wide prior is at the bound whether or not the graph relinearizes, which is
# why the knob is inert there. The tight prior is not: relinearization takes it
# from 3.79 to 2.04, closing 47% of the gap, and two full factors are still
# unclaimed. That regime is not information-limited, and it is the one place all
# session where the estimator demonstrably leaves something behind.
#
# WHAT THE Mag 4 CASES TURN ON. The bound is exactly linear in sigma -- 1.69 m
# at 1.0 nT, 7.42 at 4.40, 20.24 at 12.0, a slope of 1.687 m per nT. With the
# full-line Mag 4 smoothed DRMS at 24.19 m, ratio_s = 14.3 / sigma_Mag4, so the
# question is only whether Mag 4's post-TL residual clears 14.3 nT against
# Mag 5's 4.40. Step 1 of the audit answers it in seconds, no estimator needed.
#
# ONE HONEST SOFTNESS. sigma comes from a STATIC TL fit at the true position,
# which is what the model class cannot explain without letting the compensation
# chase the map. The estimator's own post-fit residual (2.76 nT) would read 1.69
# instead of 1.06 and would be circular. Either choice is near one, not five.
# The bound also projects a static TL subspace while the estimator carries a
# walking one, so the real nuisance space is larger and this bound is
# optimistic -- the right direction for a bound, but worth stating.
set -e
cd /mnt/c/Users/cin64/Desktop/MagNav
PY=~/gtsam_env/bin/python
DATA=~/magnav_data
G=research/gtsam_poc
CASES="1007_06:4 1007_06:5 1007_02:4 1007_02:5 1003_02:4 1003_02:5 1003_08:4 1003_08:5"
LAG=300
WARM=600

OUT=$G/bound_sweep.log
TSV=$G/bound_sweep.tsv
: > $OUT; : > $TSV
for C in $CASES; do
    LINE=${C%:*}; MAG=${C#*:}
    H5=$DATA/line_${LINE}_full.h5
    # prefer the committed run's npz, but never write over it: a regeneration
    # goes to its own _bs_ tag, so a wrong --data path or a changed default can
    # only produce a new file rather than silently replace a paper artifact
    NPZ=$G/est_gtsam_ps100_${LINE}_m${MAG}.npz
    if [ ! -f "$NPZ" ]; then
        NPZ=$G/est_gtsam_bs_${LINE}_m${MAG}.npz
        if [ ! -f "$NPZ" ]; then
            echo "=== regenerating as _bs_${LINE}_m${MAG} (ps100 npz absent) ==="
            $PY -u $G/run_gtsam_decimated.py $H5 $LAG $MAG 10 _bs_${LINE}_m${MAG} \
                --tl-sigma 100 > $G/x_bs_${LINE}_m${MAG}.log 2>&1
            grep -h "warm=600" $G/gtsam_poc_result_bs_${LINE}_m${MAG}.txt
            echo "  compare against gtsam_poc_result_ps100_${LINE}_m${MAG}.txt:"
            grep -h "warm=600" $G/gtsam_poc_result_ps100_${LINE}_m${MAG}.txt 2>/dev/null \
                || echo "  (no committed ps100 result to compare)"
        fi
    fi
    echo "=== $LINE Mag $MAG ==="
    $PY -u $G/noise_model_audit.py $H5 --mag $MAG --lag $LAG --est $NPZ \
        --warm $WARM > $G/x_audit_${LINE}_m${MAG}.log 2>&1
    cat $G/x_audit_${LINE}_m${MAG}.log >> $OUT
    grep -E "static 19-dof|fitted FOGM|<- measured|<- assumed|DRMS" \
        $G/x_audit_${LINE}_m${MAG}.log || true
    A=$(grep -m1 "^AUDIT " $G/x_audit_${LINE}_m${MAG}.log || true)
    [ -n "$A" ] && echo -e "${LINE}\tm${MAG}\t${A#AUDIT }" >> $TSV
done

echo
echo "########## achieved over the information bound ##########"
printf '%-14s %8s %8s %9s %10s %9s %8s %8s\n' \
       case "sig[nT]" "tau[s]" "bound[m]" "smoothed" "causal" "ratio_s" "ratio_c"
while IFS=$'\t' read -r LINE MAG REST; do
    eval "$REST"
    printf '%-14s %8s %8s %9s %10s %9s %8s %8s\n' \
           "$LINE $MAG" "$sigma" "$tau" "$bound" "$smoothed" "$causal" \
           "$ratio_s" "$ratio_c"
done < $TSV

echo
echo "ratio_s near 1.0: information-limited, no modelling change will help."
echo "ratio_s well above 1.0: that case is where the remaining work is."
echo "ratio_c is expected near 2 even on a finished case -- the bound is"
echo "two-sided over the lag and a causal estimate only has one side."
echo BOUND_SWEEP_DONE
