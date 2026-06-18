#!/bin/bash
# Build fresh MP calibration threshold tables for avg_stoc_len = 64/96/128,
# in the CURRENT repo (not reusing scmp_diffusion_prev's April tables).
# Runs 3 calibrations pinned to local GPUs 0/1/2. bitrev, cosine/FP-teacher,
# mp_levels 256..16, budget_ratio = target/256. Mirrors calib_sweep_targets_bitrev.sh.
# Args: <CKPT> <OUTDIR>
set -uo pipefail
CKPT="$1"; OUTDIR="$2"
REPO=/gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_diffusion
source /home/zhkangqi/miniconda3/etc/profile.d/conda.sh
conda activate qdit
export PYTHONUNBUFFERED=1 SC_OWEN_MODE=bitrev
cd "$REPO"
mkdir -p "$OUTDIR"

declare -a T=( "64:0.25" "96:0.375" "128:0.5" )
pids=(); g=0
for e in "${T[@]}"; do
    TGT=${e%:*}; BR=${e#*:}
    echo "[gpu$g] calibrate avg=$TGT budget_ratio=$BR"
    CUDA_VISIBLE_DEVICES=$g python -u scripts/calibrate_mp_thresholds.py \
        --mp_levels 256,192,128,96,64,48,32,16 \
        --budget_ratio "$BR" --budget_ref_stoc_len 256 \
        --metric cosine --teacher fp \
        --sc_prec 8 --sc_fixed_level_prec \
        --wbits 8 --abits 8 --w_sym --a_sym \
        --image-size 256 --num-sampling-steps 50 \
        --num_calib_batches 1 --num_calib_timesteps 6 \
        --timestep_buckets 4 --layer_buckets 4 \
        --teacher_cfg_scale 0.0 \
        --ckpt "$CKPT" \
        --calib_output_json "$OUTDIR/calib_fix_avg${TGT}_l256_ref192.json" \
        --calib_summary_csv "$OUTDIR/calib_fix_avg${TGT}_summary.csv" \
        > "$OUTDIR/calib_avg${TGT}.log" 2>&1 &
    pids+=($!); g=$((g+1))
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "calib worker done rc=$rc"
exit $rc
