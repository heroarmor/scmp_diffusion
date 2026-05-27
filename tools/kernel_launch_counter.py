"""Count which Triton kernels actually launch during a DiT-XL/2 run.

Wraps each kernel's ``__getitem__`` (the [grid] launch syntax) to count
invocations. Reports the histogram at process exit.
"""
import sys, os, atexit
from pathlib import Path
from collections import Counter

ROOT = Path('/gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_diffusion')
sys.path.insert(0, str(ROOT))

import scmp_kernels.sc.kernels as K

counts = Counter()
KERNEL_NAMES = [
    'build_cum_indicator_kernel', 'compute_k_table_kernel',
    'enable_matmul_tiled_kernel', 'enable_matmul_compact_dot_kernel',
    'enable_matmul_bipolar_batched_kernel',
    'fused_quant_kernel',                     # unified: bipolar/unipolar × flat/per-row
    'fused_quant_bipolar_batched_kernel',     # per-head batched (no twin)
]


def make_counter(name, orig):
    """Return a JITFunction-like proxy that counts [grid](...) launches."""
    class Counted:
        def __getitem__(self, grid):
            counts[name] += 1
            return orig[grid]
        # forward attribute access to the wrapped JITFunction
        def __getattr__(self, attr):
            return getattr(orig, attr)
    return Counted()


for name in KERNEL_NAMES:
    orig = getattr(K, name, None)
    if orig is not None:
        setattr(K, name, make_counter(name, orig))

def _summary():
    total = sum(counts.values())
    print(f"\n{'='*80}\nTriton kernel launches during this run\n{'='*80}", flush=True)
    print(f"  total launches: {total}\n", flush=True)
    if not counts:
        print("  (none fired)"); return
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        family = ("matmul" if 'enable_matmul' in name
                  else "quant"  if 'fused_quant' in name
                  else "table-build")
        print(f"  {n:>10,}  {family:<11}  {name}", flush=True)
atexit.register(_summary)


# Replay the same all-ops uniform-L=128 run we did earlier
sys.argv = [
    'quant_sc_main.py',
    '--ckpt', os.environ.get('DIT_CKPT', 'pretrained_models/DiT-XL-2-256x256.pt'),
    '--wbits', '8', '--abits', '8', '--w_sym', '--a_sym',
    '--timewise', '1.0',
    '--qklayerwise', '1.0', '--avlayerwise', '1.0',
    '--projlayerwise', '1.0', '--mlplayerwise', '1.0',
    '--inputprojlayerwise', '1.0',
    '--sc_prec', '8', '--sc_fixed_level_prec',
    '--sc_config', 'results/sc_cfg_uniform128_all.json',
    '--image-size', '256', '--num-sampling-steps', '50',
    '--cfg-scale', '4', '--batch-size', '8',
    '--results-dir', 'results/kernel_count_run',
]
os.chdir(str(ROOT))
exec(compile(open('scripts/quant_sc_main.py').read(), 'scripts/quant_sc_main.py', 'exec'))
