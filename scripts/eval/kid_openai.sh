#!/bin/bash
# ============================================================
# Compute KID on the SAME pool3 features as the OpenAI/ADM evaluator FID,
# reusing the scmp_diffusion_prev evaluator + ImageNet-256 reference, in the
# `tfeval` conda env (TF 2.15 + bundled CUDA libs). Companion to eval_openai.sh.
#
# Usage:
#   bash scripts/eval/kid_openai.sh <out_txt> <sample_npz> [<sample_npz> ...]
# Overrides: PREV, REF_NPZ, EVALUATOR
# ============================================================
set -euo pipefail

OUT_TXT="${1:?usage: kid_openai.sh <out_txt> <sample_npz> [...]}"
shift
[[ $# -ge 1 ]] || { echo "ERROR: need >=1 sample npz" >&2; exit 1; }

PREV="${PREV:-/gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_diffusion_prev}"
REF_NPZ="${REF_NPZ:-${PREV}/imagenet256_ref/VIRTUAL_imagenet256_labeled.npz}"
export EVALUATOR="${EVALUATOR:-${PREV}/Q-DiT/models/evaluations/evaluator.py}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for f in "$EVALUATOR" "$REF_NPZ" "$HERE/kid_openai.py"; do
    [[ -e "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

source /home/zhkangqi/miniconda3/etc/profile.d/conda.sh
conda activate tfeval
NV=$(python -c "import os, nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH=${NV}/cudnn/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cuda_nvrtc/lib:${NV}/cublas/lib:${NV}/cufft/lib:${NV}/curand/lib:${NV}/cusolver/lib:${NV}/cusparse/lib:${NV}/nvjitlink/lib:${LD_LIBRARY_PATH:-}
export TF_CPP_MIN_LOG_LEVEL=2

echo "=== KID: ref=${REF_NPZ} ==="
python -u "$HERE/kid_openai.py" "$REF_NPZ" "$OUT_TXT" "$@"
echo "=== done: ${OUT_TXT} ==="
cat "$OUT_TXT"
