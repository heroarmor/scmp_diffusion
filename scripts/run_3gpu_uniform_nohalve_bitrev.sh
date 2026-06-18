#!/bin/bash
# ============================================================
# 3-GPU parallel stoc_len sweep, halve=OFF, bitrev scramble.
#
#   GPU 0 -> uniform128, GPU 1 -> uniform96, GPU 2 -> uniform64.
#   Each GPU generates the full balanced NUM_FID set (10/class x 1000)
#   for its stoc_len, concurrently. sc_prec=8, halve OFF, bitrev
#   scramble, qk & av = per_row, all operators + all timesteps SC.
#   Resumable (idempotent on existing PNGs).
#
# Usage (from a >=3-GPU allocation):
#   conda activate qdit
#   cd .../scmp_diffusion
#   bash scripts/run_3gpu_uniform_nohalve_bitrev.sh
#
# Override: NUM_FID, STOC_LENS (must have <= #GPUs entries), BATCH,
#   NUM_STEPS, CFG_SCALE, OWEN_MODE, NUM_CLASSES, CKPT, OUT_BASE, CONFIG_DIR,
#   RUN_EVAL (1 to eval after generation).
# ============================================================
set -euo pipefail

NUM_FID="${NUM_FID:-10000}"
STOC_LENS="${STOC_LENS:-128,96,64}"
BATCH="${BATCH:-64}"
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
SWEEP_LOG="${OUT_BASE}/parallel_sweep.log"

(( NUM_FID % NUM_CLASSES == 0 )) || { echo "ERROR: NUM_FID must be divisible by NUM_CLASSES" >&2; exit 1; }
[[ -f "${CKPT}" ]] || { echo "ERROR: checkpoint not found: ${CKPT}" >&2; exit 1; }

IFS=',' read -ra SLS <<< "${STOC_LENS}"

export SC_OWEN_MODE="${OWEN_MODE}"
export PYTHONUNBUFFERED=1

{
echo "============================================================"
echo "3-GPU parallel stoc_len sweep started $(date)"
echo "  one GPU per stoc_len: ${STOC_LENS}"
echo "  NUM_FID=${NUM_FID}  BATCH=${BATCH}  NUM_STEPS=${NUM_STEPS}  CFG_SCALE=${CFG_SCALE}"
echo "  OWEN_MODE=${OWEN_MODE}  sc_prec=8 halve=OFF qk/av=per_row"
echo "  CKPT=${CKPT}  OUT_BASE=${OUT_BASE}"
echo "============================================================"
} | tee -a "${SWEEP_LOG}"

cd "${REPO_ROOT}"

PIDS=()
for GPU_ID in "${!SLS[@]}"; do
    SL="${SLS[$GPU_ID]}"
    TAG="uniform${SL}"
    CFG_DIR="${OUT_BASE}/${TAG}"
    SAMPLES="${CFG_DIR}/samples"
    IDX_DIR="${CFG_DIR}/_indices"
    SC_JSON="${CONFIG_DIR}/sc_cfg_uniform${SL}_all.json"
    GPU_LOG_DIR="${CFG_DIR}/_logs/gpu_${GPU_ID}"
    mkdir -p "${SAMPLES}" "${IDX_DIR}" "${GPU_LOG_DIR}"

    [[ -f "${SC_JSON}" ]] || { echo "[skip] ${TAG}: config missing ${SC_JSON}" | tee -a "${SWEEP_LOG}"; continue; }

    # All NUM_FID indices assigned to this single GPU (resumable: the runner
    # skips indices whose PNG already exists).
    INDICES_FILE="${IDX_DIR}/all.txt"
    seq 0 $((NUM_FID - 1)) > "${INDICES_FILE}"

    echo "[launch] GPU ${GPU_ID} -> ${TAG} (${NUM_FID} balanced, all on this GPU) at $(date +%H:%M:%S)" | tee -a "${SWEEP_LOG}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    SC_OWEN_MODE="${OWEN_MODE}" \
    python -u scripts/quant_sc_main.py \
        --wbits 8 --abits 8 --w_sym --a_sym \
        --timewise 1 --qklayerwise 1.0 --avlayerwise 1.0 \
        --projlayerwise 1.0 --mlplayerwise 1.0 --inputprojlayerwise 1.0 \
        --sc_prec 8 --sc_fixed_level_prec \
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

echo "[wait] ${#PIDS[@]} parallel workers: ${PIDS[*]}" | tee -a "${SWEEP_LOG}"
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  [worker $i] done at $(date +%H:%M:%S)" | tee -a "${SWEEP_LOG}"
    else
        echo "  [worker $i] FAILED (rc=$?)" | tee -a "${SWEEP_LOG}"
        FAILED=$((FAILED + 1))
    fi
done

for SL in "${SLS[@]}"; do
    N=$(find "${OUT_BASE}/uniform${SL}/samples" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)
    echo "[count] uniform${SL}: ${N}/${NUM_FID}" | tee -a "${SWEEP_LOG}"
done
echo "Generation finished $(date) (failed workers: ${FAILED})" | tee -a "${SWEEP_LOG}"

# --- robust ADM/OpenAI eval (Inception Score / FID / sFID / Precision / Recall) ---
if [[ "${RUN_EVAL:-1}" == "1" && ${FAILED} -eq 0 ]]; then
    for SL in "${SLS[@]}"; do
        D="${OUT_BASE}/uniform${SL}/samples"
        N=$(find "${D}" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)
        [[ "${N}" -ge "${NUM_FID}" ]] || { echo "[eval-skip] uniform${SL} ${N}/${NUM_FID}" | tee -a "${SWEEP_LOG}"; continue; }
        echo "[eval] uniform${SL} ..." | tee -a "${SWEEP_LOG}"
        bash "${REPO_ROOT}/scripts/eval_openai.sh" "${D}" "${OUT_BASE}/uniform${SL}" 2>&1 | tee -a "${SWEEP_LOG}"
    done
    echo "=== metrics summary ===" | tee -a "${SWEEP_LOG}"
    for SL in "${SLS[@]}"; do
        T="${OUT_BASE}/uniform${SL}.openai_eval.txt"
        [[ -f "${T}" ]] || continue
        echo "uniform${SL}:" | tee -a "${SWEEP_LOG}"
        grep -E "^(Inception Score|FID|sFID|Precision|Recall):" "${T}" | sed 's/^/  /' | tee -a "${SWEEP_LOG}"
    done
fi
echo "DONE $(date)" | tee -a "${SWEEP_LOG}"
