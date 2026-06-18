#!/bin/bash
# Runs on ONE node (inside an salloc step) for the FP baseline. Launches 3 FP
# workers pinned to local GPUs 0/1/2 on index shards BASE_SHARD+{0,1,2}, all
# writing to one shared samples dir. FP = --wbits 16 --abits 16, no SC flags
# (SC layerwise fractions default to 0 -> pure FP16 DiT-XL/2).
set -uo pipefail
IDX_DIR="$1"; BASE_SHARD="$2"; SAMPLES="$3"; LOGDIR="$4"; CKPT="$5"
CFG="$6"; STEPS="$7"; BATCH="$8"; SEED="$9"; NUM_FID="${10}"; NUM_CLASSES="${11}"
REPO=/gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_diffusion

source /home/zhkangqi/miniconda3/etc/profile.d/conda.sh
conda activate qdit
export PYTHONUNBUFFERED=1
cd "$REPO"

pids=()
for g in 0 1 2; do
    shard=$((BASE_SHARD + g))
    IDXF="$IDX_DIR/gpu_${shard}.txt"
    if [[ ! -s "$IDXF" ]]; then
        echo "[$(hostname) gpu$g] shard $shard empty -> skip"
        continue
    fi
    mkdir -p "$LOGDIR/gpu_${shard}"
    echo "[$(hostname) gpu$g] shard $shard: $(wc -l < "$IDXF") indices -> log gpu_${shard}.log"
    CUDA_VISIBLE_DEVICES=$g python -u scripts/quant_sc_main.py \
        --wbits 16 --abits 16 --w_sym --a_sym \
        --image-size 256 --num-sampling-steps "$STEPS" --cfg-scale "$CFG" \
        --batch-size "$BATCH" \
        --generate-fid-samples --balanced_classes \
        --num-classes "$NUM_CLASSES" \
        --balanced_total_samples "$NUM_FID" \
        --num-fid-samples "$NUM_FID" \
        --target_indices_path "$IDXF" \
        --samples_dir_override "$SAMPLES" \
        --seed "$SEED" \
        --results-dir "$LOGDIR/gpu_${shard}" \
        --ckpt "$CKPT" \
        > "$LOGDIR/gpu_${shard}.log" 2>&1 &
    pids+=($!)
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[$(hostname)] node_worker done (base=$BASE_SHARD) rc=$rc"
exit $rc
