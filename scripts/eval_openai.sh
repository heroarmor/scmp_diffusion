#!/bin/bash
# ============================================================
# Robust ADM/OpenAI evaluation of a sample directory, reusing the
# scmp_diffusion_prev evaluator + ImageNet-256 reference.
#
# Packs a PNG dir -> samples.npz, then runs the OpenAI evaluator
# (Inception Score / FID / sFID / Precision / Recall) in the `tfeval`
# conda env against VIRTUAL_imagenet256_labeled.npz.
#
# Usage:
#   bash scripts/eval_openai.sh <samples_dir> <out_prefix>
#     <samples_dir>  dir of NNNNNN.png (256x256)
#     <out_prefix>   writes <out_prefix>.npz and <out_prefix>.openai_eval.txt
#
# Override:
#   PREV=/gpfs/.../scmp_diffusion_prev   (evaluator + reference live here)
# ============================================================
set -euo pipefail

SAMPLES="${1:?usage: eval_openai.sh <samples_dir> <out_prefix>}"
OUT_PREFIX="${2:?usage: eval_openai.sh <samples_dir> <out_prefix>}"

PREV="${PREV:-/gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_diffusion_prev}"
EVALUATOR="${EVALUATOR:-${PREV}/Q-DiT/models/evaluations/evaluator.py}"
REF_NPZ="${REF_NPZ:-${PREV}/imagenet256_ref/VIRTUAL_imagenet256_labeled.npz}"
PACKER="${PACKER:-${PREV}/imagenet256_ref/parallel_npz.py}"

for f in "$EVALUATOR" "$REF_NPZ" "$PACKER"; do
    [[ -e "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done
[[ -d "$SAMPLES" ]] || { echo "ERROR: samples dir not found: $SAMPLES" >&2; exit 1; }

source /home/zhkangqi/miniconda3/etc/profile.d/conda.sh

NPZ="${OUT_PREFIX}.npz"
EVAL_TXT="${OUT_PREFIX}.openai_eval.txt"

# 1) pack PNGs -> npz (qdit env has numpy/PIL/tqdm)
echo "=== packing $(basename "$SAMPLES") -> ${NPZ} ==="
conda activate qdit
python -u "$PACKER" "$SAMPLES" "$NPZ"
conda deactivate

# 2) OpenAI ADM evaluator in tfeval (TF 2.15 + CUDA libs on LD_LIBRARY_PATH)
echo "=== evaluating ${NPZ} vs reference ==="
conda activate tfeval
NV=$(python -c "import os, nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH=${NV}/cudnn/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cuda_nvrtc/lib:${NV}/cublas/lib:${NV}/cufft/lib:${NV}/curand/lib:${NV}/cusolver/lib:${NV}/cusparse/lib:${NV}/nvjitlink/lib:${LD_LIBRARY_PATH:-}
export TF_CPP_MIN_LOG_LEVEL=2
python -u "$EVALUATOR" "$REF_NPZ" "$NPZ" 2>&1 | tee "$EVAL_TXT"
conda deactivate

echo "=== done: ${EVAL_TXT} ==="
grep -E "^(Inception Score|FID|sFID|Precision|Recall):" "$EVAL_TXT" || true
