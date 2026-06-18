#!/usr/bin/env python
"""
KID (Kernel Inception Distance) on the SAME Inception-V3 pool3 features that the
OpenAI/ADM evaluator.py uses for FID -- so KID is directly comparable to the
FID / sFID / IS / Precision / Recall that evaluator.py already reports.

Metric: polynomial-kernel (degree 3, coef0 1, gamma 1/d) UNBIASED MMD^2,
averaged over `n_subsets` random subsets of size `subset_size`
(torch-fidelity / clean-fid default: 100 x 1000). Reported as mean +/- std.

As a consistency check we also recompute FID from the same pool3 features; it
should match evaluator.py's FID to within float noise, proving the KID is on
the same feature space as the reported FID.

Usage:
  python kid_openai.py <ref_npz> <out_txt> <sample_npz> [<sample_npz> ...]
Env override:
  EVALUATOR=/path/to/evaluator.py   (default: scmp_diffusion_prev ADM evaluator)
"""
import os
import sys

import numpy as np
from scipy import linalg

EVALUATOR_PY = os.environ.get(
    "EVALUATOR",
    "/gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/"
    "scmp_diffusion_prev/Q-DiT/models/evaluations/evaluator.py",
)
sys.path.insert(0, os.path.dirname(EVALUATOR_PY))

import tensorflow.compat.v1 as tf  # noqa: E402
from evaluator import Evaluator  # noqa: E402

SUBSET_SIZE = 1000
N_SUBSETS = 100
SEED = 0


def polynomial_kernel(x, y):
    d = x.shape[1]
    return (x.astype(np.float64) @ y.astype(np.float64).T / d + 1.0) ** 3


def mmd2_unbiased(k_xx, k_xy, k_yy):
    m = k_xx.shape[0]
    n = k_yy.shape[0]
    sum_xx = k_xx.sum() - np.trace(k_xx)
    sum_yy = k_yy.sum() - np.trace(k_yy)
    sum_xy = k_xy.sum()
    return sum_xx / (m * (m - 1)) + sum_yy / (n * (n - 1)) - 2.0 * sum_xy / (m * n)


def compute_kid(ref, smp, subset_size=SUBSET_SIZE, n_subsets=N_SUBSETS, seed=SEED):
    rng = np.random.RandomState(seed)
    m = min(subset_size, ref.shape[0], smp.shape[0])
    vals = np.empty(n_subsets, dtype=np.float64)
    for i in range(n_subsets):
        x = ref[rng.choice(ref.shape[0], m, replace=False)]
        y = smp[rng.choice(smp.shape[0], m, replace=False)]
        vals[i] = mmd2_unbiased(
            polynomial_kernel(x, x),
            polynomial_kernel(x, y),
            polynomial_kernel(y, y),
        )
    return float(vals.mean()), float(vals.std()), m, n_subsets


def fid_from_pool3(ref, smp):
    mu1, mu2 = ref.mean(0), smp.mean(0)
    s1 = np.cov(ref, rowvar=False)
    s2 = np.cov(smp, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def main():
    if len(sys.argv) < 4:
        sys.exit("usage: kid_openai.py <ref_npz> <out_txt> <sample_npz> [...]")
    ref_npz, out_txt, sample_npzs = sys.argv[1], sys.argv[2], sys.argv[3:]

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    ev = Evaluator(tf.Session(config=config))
    print("warming up TensorFlow...", flush=True)
    ev.warmup()

    print(f"computing reference pool3 activations: {ref_npz}", flush=True)
    ref_pool3 = ev.read_activations(ref_npz)[0]
    print(f"  ref pool3 {ref_pool3.shape}", flush=True)

    lines = [
        f"# KID = poly-kernel(deg3, gamma=1/d, coef0=1) unbiased MMD^2, "
        f"{N_SUBSETS}x{SUBSET_SIZE} subsets, same pool3 features as evaluator.py FID",
        f"# ref={ref_npz}  n_ref={ref_pool3.shape[0]}",
    ]
    for npz in sample_npzs:
        print(f"computing sample pool3 activations: {npz}", flush=True)
        smp_pool3 = ev.read_activations(npz)[0]
        kid_mean, kid_std, m, ns = compute_kid(ref_pool3, smp_pool3)
        fid_chk = fid_from_pool3(ref_pool3, smp_pool3)
        line = (
            f"{os.path.basename(npz)}\t"
            f"KID {kid_mean:.6e} +/- {kid_std:.2e}\t"
            f"KIDx1e3 {kid_mean * 1e3:.4f}\t"
            f"[{ns}x{m}]\tFID_check {fid_chk:.4f}\tn {smp_pool3.shape[0]}"
        )
        print(line, flush=True)
        lines.append(line)

    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[ok] wrote {out_txt}", flush=True)


if __name__ == "__main__":
    main()
