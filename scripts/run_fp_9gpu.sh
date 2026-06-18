#!/bin/bash
# ============================================================
# FP (floating-point, no-SC, no-quant) DiT-XL/2 baseline across all 9 held
# GPUs (3 nodes x 3 GPUs). 9-way class-balanced index shard -> generate ->
# eval (FID/sFID/IS/Prec/Rec via OpenAI ADM evaluator) + KID.
#
# Distributional baseline: same seed-0 RNG + same balanced 1000-class layout as
# the SC sweeps, but a fresh 9-way split (so NOT per-index noise-paired with the
# 3-way SC runs; FID/KID are distributional so this is fine).
#
# Override: CFG STEPS BATCH SEED NUM_FID NUM_CLASSES OUT_BASE JOBS
# ============================================================
set -uo pipefail
REPO=/gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_diffusion
SCRATCH=/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi
CKPT="${CKPT:-$SCRATCH/pretrained_models/DiT-XL-2-256x256.pt}"

CFG="${CFG:-1.5}"; STEPS="${STEPS:-50}"; BATCH="${BATCH:-64}"
SEED="${SEED:-0}"; NUM_FID="${NUM_FID:-10000}"; NUM_CLASSES="${NUM_CLASSES:-1000}"
OUT_BASE="${OUT_BASE:-$SCRATCH/scmp_diffusion_fid_fp_cfg${CFG/./}}"
read -ra JOBS <<< "${JOBS:-51660970 51660978 51660979}"   # gl1802 gl1804 gl1805

SAMPLES="$OUT_BASE/fp/samples"; IDX_DIR="$OUT_BASE/fp/_indices"; LOGDIR="$OUT_BASE/fp/_logs"
mkdir -p "$SAMPLES" "$IDX_DIR" "$LOGDIR"
ALIGN=$((NUM_FID / NUM_CLASSES))

echo "============================================================"
echo "FP baseline  CFG=$CFG STEPS=$STEPS BATCH=$BATCH SEED=$SEED NUM_FID=$NUM_FID"
echo "  OUT_BASE=$OUT_BASE"
echo "  JOBS=${JOBS[*]}  (9 GPUs)   CKPT=$CKPT"
echo "============================================================"
[[ -f "$CKPT" ]] || { echo "ERROR: missing CKPT $CKPT" >&2; exit 1; }

source /home/zhkangqi/miniconda3/etc/profile.d/conda.sh; conda activate qdit
echo "=== plan 9 shards (align=$ALIGN) ==="
python -u "$REPO/scripts/_plan_missing_indices.py" "$SAMPLES" "$NUM_FID" 9 "$IDX_DIR" "$ALIGN"

echo "=== dispatch 3 nodes x 3 GPUs at $(date +%H:%M:%S) ==="
spids=()
for n in 0 1 2; do
    base=$((3 * n))
    srun --jobid="${JOBS[$n]}" --overlap --gres=gpu:3 \
        bash "$REPO/scripts/_fp_node_worker.sh" \
            "$IDX_DIR" "$base" "$SAMPLES" "$LOGDIR" "$CKPT" \
            "$CFG" "$STEPS" "$BATCH" "$SEED" "$NUM_FID" "$NUM_CLASSES" \
        > "$LOGDIR/node_${n}.log" 2>&1 &
    spids+=($!)
    echo "  node $n (job ${JOBS[$n]}) shards $base..$((base+2)) dispatched"
done
grc=0
for p in "${spids[@]}"; do wait "$p" || grc=1; done
FINAL=$(find "$SAMPLES" -maxdepth 1 -name '[0-9][0-9][0-9][0-9][0-9][0-9].png' | wc -l)
echo "=== generation done rc=$grc : $FINAL/$NUM_FID samples at $(date +%H:%M:%S) ==="
if [[ "$FINAL" -lt "$NUM_FID" ]]; then
    echo "[partial] re-run this script to resume (idempotent on existing PNGs)"; exit 1
fi

echo "=== eval: FID/sFID/IS/Precision/Recall ==="
srun --jobid="${JOBS[0]}" --overlap --gres=gpu:1 \
    bash "$REPO/scripts/eval_openai.sh" "$SAMPLES" "$OUT_BASE/fp" \
    > "$LOGDIR/eval_openai.log" 2>&1
echo "=== eval: KID (same pool3 features) ==="
srun --jobid="${JOBS[0]}" --overlap --gres=gpu:1 \
    bash "$REPO/scripts/eval/kid_openai.sh" "$OUT_BASE/fp.kid.txt" "$OUT_BASE/fp.npz" \
    > "$LOGDIR/eval_kid.log" 2>&1

echo "============================================================"
echo "FP BASELINE RESULTS (CFG=$CFG)"
grep -E "^(Inception Score|FID|sFID|Precision|Recall):" "$OUT_BASE/fp.openai_eval.txt" 2>/dev/null
cat "$OUT_BASE/fp.kid.txt" 2>/dev/null
echo "DONE $(date)"
