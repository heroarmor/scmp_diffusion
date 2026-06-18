#!/bin/bash
# ============================================================
# Generalized 9-GPU (3 nodes x 3 GPU) no-SC baseline runner.
#   WBITS=16 ABITS=16 -> FP16 baseline
#   WBITS=8  ABITS=8  -> W8A8 quantized, NO stochastic computing
# Same seed-0 RNG + balanced 1000-class layout as the SC sweeps (distributional
# baseline; fresh 9-way split, not per-index noise-paired). Generate -> eval
# (FID/sFID/IS/Prec/Rec) + KID.
#
# Override: WBITS ABITS CFG STEPS BATCH SEED NUM_FID NUM_CLASSES TAG OUT_BASE JOBS
# ============================================================
set -uo pipefail
REPO=/gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_diffusion
SCRATCH=/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi
CKPT="${CKPT:-$SCRATCH/pretrained_models/DiT-XL-2-256x256.pt}"

WBITS="${WBITS:-8}"; ABITS="${ABITS:-8}"
CFG="${CFG:-1.5}"; STEPS="${STEPS:-50}"; BATCH="${BATCH:-64}"
SEED="${SEED:-0}"; NUM_FID="${NUM_FID:-10000}"; NUM_CLASSES="${NUM_CLASSES:-1000}"
TAG="${TAG:-w${WBITS}a${ABITS}nosc}"
OUT_BASE="${OUT_BASE:-$SCRATCH/scmp_diffusion_fid_${TAG}_cfg${CFG/./}}"
read -ra JOBS <<< "${JOBS:-51660970 51660978 51660979}"

SAMPLES="$OUT_BASE/samples"; IDX_DIR="$OUT_BASE/_indices"; LOGDIR="$OUT_BASE/_logs"
mkdir -p "$SAMPLES" "$IDX_DIR" "$LOGDIR"
ALIGN=$((NUM_FID / NUM_CLASSES))

echo "============================================================"
echo "$TAG baseline  w${WBITS}a${ABITS} NO-SC  CFG=$CFG STEPS=$STEPS BATCH=$BATCH SEED=$SEED NUM_FID=$NUM_FID"
echo "  OUT_BASE=$OUT_BASE   JOBS=${JOBS[*]} (9 GPUs)"
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
        bash "$REPO/scripts/_quant_node_worker.sh" \
            "$IDX_DIR" "$base" "$SAMPLES" "$LOGDIR" "$CKPT" \
            "$CFG" "$STEPS" "$BATCH" "$SEED" "$NUM_FID" "$NUM_CLASSES" "$WBITS" "$ABITS" \
        > "$LOGDIR/node_${n}.log" 2>&1 &
    spids+=($!)
    echo "  node $n (job ${JOBS[$n]}) shards $base..$((base+2)) dispatched"
done
grc=0
for p in "${spids[@]}"; do wait "$p" || grc=1; done
FINAL=$(find "$SAMPLES" -maxdepth 1 -name '[0-9][0-9][0-9][0-9][0-9][0-9].png' | wc -l)
echo "=== generation done rc=$grc : $FINAL/$NUM_FID at $(date +%H:%M:%S) ==="
if [[ "$FINAL" -lt "$NUM_FID" ]]; then echo "[partial] re-run to resume"; exit 1; fi

echo "=== eval: FID/sFID/IS/Precision/Recall ==="
srun --jobid="${JOBS[0]}" --overlap --gres=gpu:1 \
    bash "$REPO/scripts/eval_openai.sh" "$SAMPLES" "$OUT_BASE/${TAG}" \
    > "$LOGDIR/eval_openai.log" 2>&1
echo "=== eval: KID ==="
srun --jobid="${JOBS[0]}" --overlap --gres=gpu:1 \
    bash "$REPO/scripts/eval/kid_openai.sh" "$OUT_BASE/${TAG}.kid.txt" "$OUT_BASE/${TAG}.npz" \
    > "$LOGDIR/eval_kid.log" 2>&1

echo "============================================================"
echo "$TAG RESULTS (w${WBITS}a${ABITS} no-SC, CFG=$CFG)"
grep -E "^(Inception Score|FID|sFID|Precision|Recall):" "$OUT_BASE/${TAG}.openai_eval.txt" 2>/dev/null
cat "$OUT_BASE/${TAG}.kid.txt" 2>/dev/null
echo "DONE $(date)"
