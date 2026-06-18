#!/bin/bash
# ============================================================
# 3-GPU uniform-precision FID sample sweep.
#
#   sc_prec=8, halve ON (sweepable: explicit stoc_len, rng grid -> 128),
#   bitrev scramble, qk & av granularity = per_row, all operators + all
#   timesteps SC. Sweeps stoc_len in {256, 192, 128}; NUM_FID samples per
#   config, fanned out across 3 GPUs with work-stealing (resumable).
#
# Mirrors scmp_diffusion_prev/Q-DiT/scripts/run_2gpu_calib_fid_sweep.sh.
#
# Usage (from a 3-GPU allocation):
#   conda activate qdit
#   cd .../scmp_diffusion
#   bash scripts/run_3gpu_uniform_halve_bitrev.sh
#
# Override knobs (env vars):
#   NUM_FID=10000     samples per config (must be divisible by NUM_CLASSES)
#   STOC_LENS=256,192,128
#   BATCH=32          per-GPU batch
#   NUM_STEPS=50      sampling steps
#   CFG_SCALE=4       classifier-free guidance scale
#   OWEN_MODE=bitrev  counter | bitrev | random
#   CKPT=/scratch/.../pretrained_models/DiT-XL-2-256x256.pt
#   OUT_BASE=/scratch/.../scmp_diffusion_fid_halve_bitrev
#   CONFIG_DIR=$OUT_BASE/configs   (uniform${SL}_all.json live here)
#   IMAGENET_REF=/path/to/val      if set + pytorch_fid present -> compute FID
# ============================================================

set -euo pipefail

NUM_GPUS="${NUM_GPUS:-3}"
NUM_FID="${NUM_FID:-10000}"
STOC_LENS="${STOC_LENS:-256,192,128}"
BATCH="${BATCH:-32}"
NUM_STEPS="${NUM_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-4}"
OWEN_MODE="${OWEN_MODE:-bitrev}"
NUM_CLASSES="${NUM_CLASSES:-1000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi}"
CKPT="${CKPT:-${SCRATCH_ROOT}/pretrained_models/DiT-XL-2-256x256.pt}"
OUT_BASE="${OUT_BASE:-${SCRATCH_ROOT}/scmp_diffusion_fid_halve_bitrev}"
CONFIG_DIR="${CONFIG_DIR:-${OUT_BASE}/configs}"

mkdir -p "${OUT_BASE}"
SWEEP_LOG="${OUT_BASE}/sweep.log"

if (( NUM_FID % NUM_CLASSES != 0 )); then
    echo "ERROR: NUM_FID (${NUM_FID}) must be divisible by NUM_CLASSES (${NUM_CLASSES})" >&2
    exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}" >&2
    exit 1
fi

export SC_OWEN_MODE="${OWEN_MODE}"
export PYTHONUNBUFFERED=1

{
echo "============================================================"
echo "3-GPU uniform halve+bitrev sweep started $(date)"
echo "  NUM_GPUS=${NUM_GPUS}  NUM_FID=${NUM_FID}  STOC_LENS=${STOC_LENS}"
echo "  BATCH=${BATCH}  NUM_STEPS=${NUM_STEPS}  CFG_SCALE=${CFG_SCALE}"
echo "  OWEN_MODE=${OWEN_MODE}  sc_prec=8 halve=ON qk/av=per_row"
echo "  CKPT=${CKPT}"
echo "  OUT_BASE=${OUT_BASE}"
echo "============================================================"
} | tee -a "${SWEEP_LOG}"

cd "${REPO_ROOT}"

run_config() {
    local SL="$1"
    local TAG="uniform${SL}"
    local CFG_DIR="${OUT_BASE}/${TAG}"
    local SAMPLES="${CFG_DIR}/samples"
    local IDX_DIR="${CFG_DIR}/_indices"
    local SC_JSON="${CONFIG_DIR}/sc_cfg_uniform${SL}_all.json"
    mkdir -p "${SAMPLES}" "${IDX_DIR}"

    if [[ ! -f "${SC_JSON}" ]]; then
        echo "[skip] ${TAG}: config not found ${SC_JSON}" | tee -a "${SWEEP_LOG}"
        return
    fi

    local DONE
    DONE=$(find "${SAMPLES}" -maxdepth 1 -type f -name '[0-9][0-9][0-9][0-9][0-9][0-9].png' 2>/dev/null | wc -l)
    if [[ ${DONE} -ge ${NUM_FID} ]]; then
        echo "[skip] ${TAG} already complete (${DONE}/${NUM_FID})" | tee -a "${SWEEP_LOG}"
        return
    fi

    echo "[plan] ${TAG} ${DONE}/${NUM_FID} done; planning across ${NUM_GPUS} GPUs at $(date +%H:%M:%S)" | tee -a "${SWEEP_LOG}"
    python -u "${REPO_ROOT}/scripts/_plan_missing_indices.py" \
        "${SAMPLES}" "${NUM_FID}" "${NUM_GPUS}" "${IDX_DIR}" | tee -a "${SWEEP_LOG}"

    local PIDS=()
    for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
        local INDICES_FILE="${IDX_DIR}/gpu_${GPU_ID}.txt"
        if [[ ! -s "${INDICES_FILE}" ]]; then
            echo "  [GPU ${GPU_ID}] no work assigned" | tee -a "${SWEEP_LOG}"
            continue
        fi
        local GPU_LOG_DIR="${CFG_DIR}/_logs/gpu_${GPU_ID}"
        mkdir -p "${GPU_LOG_DIR}"
        CUDA_VISIBLE_DEVICES=${GPU_ID} \
        SC_OWEN_MODE="${OWEN_MODE}" \
        python -u scripts/quant_sc_main.py \
            --wbits 8 --abits 8 --w_sym --a_sym \
            --timewise 1 --qklayerwise 1.0 --avlayerwise 1.0 \
            --projlayerwise 1.0 --mlplayerwise 1.0 --inputprojlayerwise 1.0 \
            --sc_prec 8 --sc_fixed_level_prec --sc_halve \
            --sc_qk_granularity per_row \
            --sc_config "${SC_JSON}" \
            --image-size 256 --num-sampling-steps "${NUM_STEPS}" --cfg-scale "${CFG_SCALE}" \
            --batch-size "${BATCH}" \
            --generate-fid-samples \
            --balanced_classes \
            --num-classes "${NUM_CLASSES}" \
            --balanced_total_samples "${NUM_FID}" \
            --num-fid-samples "${NUM_FID}" \
            --target_indices_path "${INDICES_FILE}" \
            --samples_dir_override "${SAMPLES}" \
            --seed ${GPU_ID} \
            --results-dir "${GPU_LOG_DIR}" \
            --ckpt "${CKPT}" \
            > "${GPU_LOG_DIR}/run.log" 2>&1 &
        PIDS+=($!)
        sleep 5
    done

    local FAILED=0
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "  [worker $i] done at $(date +%H:%M:%S)" | tee -a "${SWEEP_LOG}"
        else
            echo "  [worker $i] FAILED (rc=$?); see ${CFG_DIR}/_logs/" | tee -a "${SWEEP_LOG}"
            FAILED=$((FAILED + 1))
        fi
    done

    local FINAL
    FINAL=$(find "${SAMPLES}" -maxdepth 1 -type f -name '[0-9][0-9][0-9][0-9][0-9][0-9].png' 2>/dev/null | wc -l)
    if [[ ${FINAL} -ge ${NUM_FID} ]]; then
        echo "[ok]   ${TAG} ${FINAL}/${NUM_FID} samples in ${SAMPLES} at $(date +%H:%M:%S)" | tee -a "${SWEEP_LOG}"
    else
        echo "[partial] ${TAG} ${FINAL}/${NUM_FID}; rerun this script to resume" | tee -a "${SWEEP_LOG}"
    fi
}

IFS=',' read -ra SLS <<< "${STOC_LENS}"
for SL in "${SLS[@]}"; do
    run_config "${SL}"
done

echo "Sample generation finished $(date)" | tee -a "${SWEEP_LOG}"

# --- robust ADM/OpenAI evaluation (reuses scmp_diffusion_prev evaluator +
#     ImageNet-256 reference): Inception Score / FID / sFID / Precision / Recall.
#     Set RUN_EVAL=0 to skip (e.g. generate now, evaluate later).
if [[ "${RUN_EVAL:-1}" == "1" ]]; then
    for SL in "${SLS[@]}"; do
        D="${OUT_BASE}/uniform${SL}/samples"
        N=$(find "${D}" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)
        if [[ "${N}" -lt "${NUM_FID}" ]]; then
            echo "[eval-skip] uniform${SL}: only ${N}/${NUM_FID} samples" | tee -a "${SWEEP_LOG}"
            continue
        fi
        echo "[eval] uniform${SL} ($(date +%H:%M:%S)) ..." | tee -a "${SWEEP_LOG}"
        bash "${REPO_ROOT}/scripts/eval_openai.sh" "${D}" "${OUT_BASE}/uniform${SL}" \
            2>&1 | tee -a "${SWEEP_LOG}"
    done
    echo "=== metrics summary ===" | tee -a "${SWEEP_LOG}"
    for SL in "${SLS[@]}"; do
        T="${OUT_BASE}/uniform${SL}.openai_eval.txt"
        [[ -f "${T}" ]] || continue
        echo "uniform${SL}:" | tee -a "${SWEEP_LOG}"
        grep -E "^(Inception Score|FID|sFID|Precision|Recall):" "${T}" | sed 's/^/  /' | tee -a "${SWEEP_LOG}"
    done
else
    echo "Eval skipped (RUN_EVAL=0). Run scripts/eval_openai.sh per config later." | tee -a "${SWEEP_LOG}"
fi
echo "DONE $(date)" | tee -a "${SWEEP_LOG}"
