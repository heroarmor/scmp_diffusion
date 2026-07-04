"""
Generate samples from a quantized DiT model with stochastic computing support.

This script extends quant_main.py to support SC-based matrix multiplication
for attention operations, controlled by timewise and layerwise parameters.

Usage:
    python scripts/quant_sc_main.py \
        --wbits 8 --abits 8 \
        --w_sym --a_sym \
        --timewise 0.5 \
        --qklayerwise 0.25 \
        --sc_prec 8 \
        --image-size 256
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import os
import signal
import numpy as np
import torch
import logging
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm
import math

# Graceful shutdown flag — set by SIGINT/SIGTERM handler
_shutdown_requested = False

def _shutdown_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n[Signal {signum}] Shutdown requested. Will exit after current batch...")

signal.signal(signal.SIGINT, _shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)

from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from utils.download import find_model
from models.models import DiT_models
from utils.logger_setup import create_logger
from glob import glob
from copy import deepcopy

from qdit.quant import *
from qdit.outlier import *
from qdit.datautils import *
from collections import defaultdict

from qdit.sc_integration import (
    SCController,
    MPConfig,
    add_sc_wrapper,
    quantize_sc_model,
    create_sc_controller_from_args,
)
from qdit.sc_integration.mp_config import AdaptiveMPConfig, RangeMPConfig

from torch.cuda.amp import autocast


class SCDiffusionWrapper:
    """
    Wrapper for diffusion sampling that integrates SC controller.

    Updates the SC controller's timestep before each denoising step.
    """

    def __init__(self, diffusion, sc_controller: SCController):
        self.diffusion = diffusion
        self.sc_controller = sc_controller

    def ddim_sample_loop(self, model, shape, noise, clip_denoised,
                         model_kwargs, progress, device):
        """
        DDIM sampling loop with SC controller integration.
        """
        # Get the underlying sample loop generator
        indices = list(range(self.diffusion.num_timesteps))[::-1]

        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)

        if progress:
            from tqdm.auto import tqdm as tqdm_auto
            indices = tqdm_auto(indices)

        # Nsight profiling: set NSIGHT_PROFILE_TIMESTEP=50 to profile only the 50th step
        nsight_step = os.environ.get("NSIGHT_PROFILE_TIMESTEP", None)
        nsight_step = int(nsight_step) if nsight_step is not None else None

        for step_idx, i in enumerate(indices):
            # Update SC controller with current timestep
            self.sc_controller.set_timestep(i)

            if nsight_step is not None and step_idx == nsight_step:
                torch.cuda.cudart().cudaProfilerStart()

            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.diffusion.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
                img = out["sample"]

            if nsight_step is not None and step_idx == nsight_step:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()

        return img


def validate_model_sc(args, model, diffusion, vae, sc_controller):
    """Validate model with SC support."""
    seed_everything(args.seed)
    device = next(model.parameters()).device
    using_cfg = args.cfg_scale > 1.0

    # Labels to condition the model with
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    n = len(class_labels)
    z = torch.randn(n, 4, model.input_size, model.input_size, device=device)
    y = torch.tensor(class_labels, device=device)

    if using_cfg:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    else:
        model_kwargs = dict(y=y)

    z = z.half()

    # Create SC-aware diffusion wrapper
    sc_diffusion = SCDiffusionWrapper(diffusion, sc_controller)

    with autocast():
        samples = sc_diffusion.ddim_sample_loop(
            model, z.shape, z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device
        )

    if using_cfg:
        samples, _ = samples.chunk(2, dim=0)

    samples = vae.decode(samples / 0.18215).sample
    save_image(
        samples,
        f'{args.experiment_dir}/sample_sc.png',
        nrow=4, normalize=True, value_range=(-1, 1)
    )
    print("Finish validating SC samples!")


def generate_samples_for_fid(args, model, diffusion, vae, sc_controller):
    """Generate samples for FID evaluation."""
    # Use a local Generator for z/y sampling only — do NOT touch the global
    # RNG state so that SC internal randomness is unaffected.
    sample_rng = torch.Generator(device="cpu").manual_seed(args.seed)
    cuda_rng = torch.Generator(device="cuda").manual_seed(args.seed)

    device = next(model.parameters()).device
    using_cfg = args.cfg_scale > 1.0

    # Create sample directory. --samples_dir_override lets the sweep wrapper
    # point all GPUs at a single shared dir, with global indices in filenames.
    sample_dir = getattr(args, 'samples_dir_override', None) or f'{args.experiment_dir}/samples'
    os.makedirs(sample_dir, exist_ok=True)

    # Create SC-aware diffusion wrapper
    sc_diffusion = SCDiffusionWrapper(diffusion, sc_controller)

    num_samples = args.num_fid_samples
    batch_size = args.batch_size
    num_batches = math.ceil(num_samples / batch_size)

    # ----- index-driven mode: this process produces exactly the listed -----
    # global indices and writes them to {sample_dir}/{idx:06d}.png. The sweep
    # wrapper runs this mode with NUM_GPUS workers all writing to one shared
    # sample_dir; partitioning across an arbitrary GPU count is the planner's
    # job, not this loop's.
    target_indices_path = getattr(args, 'target_indices_path', None)
    if target_indices_path is not None:
        if not getattr(args, 'balanced_classes', False):
            raise ValueError("--target_indices_path requires --balanced_classes")
        total_for_balance = getattr(args, 'balanced_total_samples', None) or num_samples
        spc = max(1, total_for_balance // args.num_classes)
        labels_global = []
        for c in range(args.num_classes):
            labels_global.extend([c] * spc)
        while len(labels_global) < total_for_balance:
            labels_global.extend(list(range(args.num_classes)))
        labels_global = labels_global[:total_for_balance]

        with open(target_indices_path) as f:
            target_indices = [int(line.strip()) for line in f if line.strip()]
        if not target_indices:
            print(f"[target] {target_indices_path} is empty; nothing to do.")
            torch.cuda.empty_cache()
            return
        out_of_range = [idx for idx in target_indices if idx < 0 or idx >= total_for_balance]
        if out_of_range:
            raise ValueError(
                f"target indices out of range [0, {total_for_balance}): "
                f"first 5 offenders = {out_of_range[:5]}")

        # Idempotent: skip any index whose final PNG already exists. Allows a
        # killed run to be resumed simply by relaunching with the same index file.
        todo = [idx for idx in target_indices
                if not os.path.exists(f"{sample_dir}/{idx:06d}.png")]
        already_done = len(target_indices) - len(todo)
        if not todo:
            print(f"[target] all {len(target_indices)} listed indices already on disk")
            torch.cuda.empty_cache()
            return
        print(f"[target] assigned {len(target_indices)} indices, "
              f"{already_done} already done, {len(todo)} to generate")

        for batch_start in tqdm(range(0, len(todo), batch_size),
                                desc="Generating samples (target_indices)"):
            if _shutdown_requested:
                print(f"Shutdown requested. Stopping after {batch_start} of {len(todo)}.")
                break
            batch_global_idx = todo[batch_start : batch_start + batch_size]
            B = len(batch_global_idx)
            y = torch.tensor(
                [labels_global[idx] for idx in batch_global_idx],
                device=device, dtype=torch.long,
            )
            z = torch.randn(B, 4, model.input_size, model.input_size,
                            device=device, generator=cuda_rng)

            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * B, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            else:
                model_kwargs = dict(y=y)

            z = z.half()
            with autocast():
                samples = sc_diffusion.ddim_sample_loop(
                    model, z.shape, z,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=True,
                    device=device,
                )
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)

            samples = vae.decode(samples / 0.18215).sample
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()

            # Atomic write (.tmp + rename) so a kill mid-save never leaves a
            # half-written PNG that a future scan would mistake for done.
            for global_idx, img in zip(batch_global_idx, samples):
                final_path = f"{sample_dir}/{global_idx:06d}.png"
                tmp_path = f"{final_path}.tmp.{os.getpid()}"
                Image.fromarray(img).save(tmp_path, format="PNG")
                os.replace(tmp_path, final_path)

        print(f"[target] finished pid={os.getpid()} samples_dir={sample_dir}")
        torch.cuda.empty_cache()
        return

    # Optional class-balanced sampling.
    #
    # Build a global label array of length `balanced_total_samples` that has
    # exactly `samples_per_class = balanced_total_samples / num_classes` images
    # per class, contiguous: [0,0,...,1,1,...,...,999,999,...].  Each process
    # then takes the slice `[class_start_idx : class_start_idx + num_samples]`.
    #
    # Single-GPU (no override): balanced_total_samples = num_samples → standard
    # 10-per-class for FID-10k.
    # Multi-GPU: pass `--balanced_total_samples 10000` and `--class_start_idx
    # GPU_ID*samples_per_gpu` so the union of all GPUs' slices is a complete
    # 10/class × 1000-class layout.
    if getattr(args, 'balanced_classes', False):
        total_for_balance = getattr(args, 'balanced_total_samples', None) or num_samples
        if total_for_balance < num_samples:
            raise ValueError(
                f"--balanced_total_samples ({total_for_balance}) must be >= "
                f"--num-fid-samples ({num_samples})")
        spc = max(1, total_for_balance // args.num_classes)
        labels_global = []
        for c in range(args.num_classes):
            labels_global.extend([c] * spc)
        # If total_for_balance isn't divisible by num_classes, top up by cycling.
        while len(labels_global) < total_for_balance:
            labels_global.extend(list(range(args.num_classes)))
        labels_global = labels_global[:total_for_balance]
        end_idx = args.class_start_idx + num_samples
        if end_idx > total_for_balance:
            raise ValueError(
                f"class_start_idx + num_samples = {end_idx} exceeds "
                f"balanced_total_samples = {total_for_balance}")
        balanced_labels = labels_global[args.class_start_idx:end_idx]
        print(f"Class-balanced: {spc} samples/class globally over "
              f"{total_for_balance} total samples; this process labels "
              f"classes {balanced_labels[0]}..{balanced_labels[-1]} "
              f"(start_idx={args.class_start_idx}, n={num_samples}).")
    else:
        balanced_labels = None

    sample_count = 0
    if getattr(args, 'resume', False):
        existing_pngs = glob(f"{sample_dir}/" + "[0-9]" * 6 + ".png")
        sample_count = len(existing_pngs)
        if sample_count >= num_samples:
            print(f"[resume] {sample_dir} already has {sample_count}/{num_samples} samples; nothing to do.")
            torch.cuda.empty_cache()
            return
        if sample_count > 0:
            print(f"[resume] continuing from sample {sample_count}/{num_samples}")
    print(f"Generating {num_samples - sample_count} samples for FID evaluation...")

    for batch_idx in tqdm(range(num_batches), desc="Generating samples"):
        if _shutdown_requested:
            print(f"Shutdown requested. Stopping after {sample_count} samples.")
            break

        # Determine actual batch size for last batch
        current_batch_size = min(batch_size, num_samples - sample_count)
        if current_batch_size <= 0:
            break

        # Class labels: balanced (sequential per class) or random
        if balanced_labels is not None:
            y = torch.tensor(
                balanced_labels[sample_count : sample_count + current_batch_size],
                device=device, dtype=torch.long,
            )
        else:
            y = torch.randint(0, args.num_classes, (current_batch_size,),
                              device=device, generator=cuda_rng)
        z = torch.randn(current_batch_size, 4, model.input_size, model.input_size,
                         device=device, generator=cuda_rng)

        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * current_batch_size, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        else:
            model_kwargs = dict(y=y)

        z = z.half()

        with autocast():
            samples = sc_diffusion.ddim_sample_loop(
                model, z.shape, z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                device=device
            )

        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)

        # Decode and save individual images
        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()

        for i, sample in enumerate(samples):
            img_path = f'{sample_dir}/{sample_count:06d}.png'
            Image.fromarray(sample).save(img_path)
            sample_count += 1
            if sample_count >= num_samples:
                break

    print(f"Generated {sample_count} samples in {sample_dir}")

    # Explicitly release GPU memory
    torch.cuda.empty_cache()


def main():
    args = create_argparser().parse_args()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}')

    # Setup experiment folder
    os.makedirs(args.results_dir, exist_ok=True)
    quant_method = "qdit_sc"
    quant_string_name = f"{quant_method}_w{args.wbits}a{args.abits}_t{args.timewise}_qk{args.qklayerwise}"
    if args.avlayerwise > 0:
        quant_string_name += f"_av{args.avlayerwise}"
    if args.projlayerwise > 0:
        quant_string_name += f"_proj{args.projlayerwise}"
    if args.mlplayerwise > 0:
        quant_string_name += f"_mlp{args.mlplayerwise}"
    if args.inputprojlayerwise > 0:
        quant_string_name += f"_inproj{args.inputprojlayerwise}"
    if args.sc_fixed_level_prec:
        quant_string_name += "_fixlvlprec"
    if args.sc_noise_model:
        quant_string_name += "_noisemodel"
    # Append per-operator MP alpha/beta to folder name when adaptive_mp is used
    if getattr(args, 'adaptive_mp', False) or getattr(args, 'adaptive_mp_table', None):
        _mp_parts = []
        for _op in ["qk", "av", "proj", "input_proj", "mlp_fc1", "mlp_fc2"]:
            _a = getattr(args, f"mp_alpha_{_op}", None)
            _b = getattr(args, f"mp_beta_{_op}", None)
            if _a is not None and _b is not None:
                _mp_parts.append(f"{_op}_a{_a}_b{_b}")
        if getattr(args, 'adaptive_mp_table', None):
            quant_string_name += f"_mptbl_{Path(args.adaptive_mp_table).stem}"
        elif _mp_parts:
            quant_string_name += "_mp_" + "_".join(_mp_parts)
        else:
            quant_string_name += f"_mp_a{args.mp_alpha}_b{args.mp_beta}"
    if getattr(args, 'range_mp', False):
        _rmp_parts = []
        for _op in ["qk", "av", "proj", "input_proj", "mlp"]:
            _t = getattr(args, f"range_mp_threshold_{_op}", None)
            if _t is not None:
                _rmp_parts.append(f"{_op}_t{_t}")
        if _rmp_parts:
            quant_string_name += "_rmp_" + "_".join(_rmp_parts)
        else:
            quant_string_name += f"_rmp_t{args.range_mp_threshold}"
    existing = sorted(glob(f"{args.results_dir}/*"))
    experiment_index = 0
    matching_dirs = []
    for p in existing:
        name = os.path.basename(p)
        if name[:3].isdigit():
            experiment_index = max(experiment_index, int(name[:3]) + 1)
            if name[4:] == quant_string_name and os.path.isdir(p):
                matching_dirs.append(p)
    if getattr(args, 'resume', False) and matching_dirs:
        experiment_dir = matching_dirs[-1]
        print(f"[resume] reusing existing experiment_dir: {experiment_dir}")
    else:
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{quant_string_name}"
    args.experiment_dir = experiment_dir
    os.makedirs(experiment_dir, exist_ok=True)

    create_logger(experiment_dir)
    logging.info(f"Experiment directory: {experiment_dir}")
    logging.info(f"SC Parameters: timewise={args.timewise}, qklayerwise={args.qklayerwise}, "
                 f"avlayerwise={args.avlayerwise}, projlayerwise={args.projlayerwise}, "
                 f"mlplayerwise={args.mlplayerwise}, inputprojlayerwise={args.inputprojlayerwise}, "
                 f"sc_prec={args.sc_prec}, sc_fixed_level_prec={args.sc_fixed_level_prec}, "
                 f"sc_noise_model={args.sc_noise_model}")
    logging.info(f"Quant Parameters: wbits={args.wbits}, abits={args.abits}, w_sym={args.w_sym}, a_sym={args.a_sym}")

    # Load model
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Process group size arguments
    args.weight_group_size = eval(args.weight_group_size)
    args.act_group_size = eval(args.act_group_size)
    args.weight_group_size = [args.weight_group_size] * len(model.blocks) if isinstance(args.weight_group_size, int) else args.weight_group_size
    args.act_group_size = [args.act_group_size] * len(model.blocks) if isinstance(args.act_group_size, int) else args.act_group_size

    # Create SC controller
    sc_controller = create_sc_controller_from_args(args, model)
    if args.debug_sc:
        sc_controller.enable_debug()
        print("DEBUG MODE: SC operators will compute both FP and SC, log errors, use FP downstream.")
    logging.info(f"SC Controller: {sc_controller}")

    # Get activation scales for static quantization
    print("Getting activation stats...")
    if args.static:
        dataloader = get_loader(args.calib_data_path, nsamples=1024, batch_size=16)
        scales = get_act_scales(model, diffusion, dataloader, device, args)
    else:
        scales = defaultdict(lambda: None)

    # Add SC wrapper (converts to SCDiTBlock)
    print("Adding SC wrappers...")
    model = add_sc_wrapper(model, device, args, scales, sc_controller)

    # Initialize range-based MP before quantization (needs to be set before
    # quantize_sc_model so it can compute per-group stoc_lens from weight ranges)
    if args.range_mp:
        range_levels = [int(x) for x in args.range_mp_levels.split(',')]
        range_op_thresholds = {}
        for op, group_key in [("qk", "qk"), ("av", "av"),
                               ("mlp_fc1", "mlp"), ("mlp_fc2", "mlp"),
                               ("input_proj", "input_proj"), ("proj", "proj")]:
            t = getattr(args, f"range_mp_threshold_{op}", None)
            if t is None:
                group_fallback = {"mlp_fc1": "mlp", "mlp_fc2": "mlp"}.get(op)
                if group_fallback:
                    t = getattr(args, f"range_mp_threshold_{group_fallback}", None)
            if t is not None:
                range_op_thresholds[op] = t
        range_config = RangeMPConfig(
            stoc_len_levels=range_levels,
            base_threshold=args.range_mp_threshold,
            operator_thresholds=range_op_thresholds,
        )
        sc_controller.init_range_mp(range_config)
        logging.info(f"Range-based MP enabled: levels={range_levels}, "
                     f"threshold={args.range_mp_threshold}, "
                     f"operator_thresholds={range_op_thresholds}")

    # Quantize weights
    print("Quantizing weights...")
    model = quantize_sc_model(model, device, args, sc_controller=sc_controller)

    # Initialize mixed precision if requested
    if args.adaptive_mp or args.adaptive_mp_table:
        levels = [int(x) for x in args.mp_levels.split(',')]
        # Build per-operator overrides (None means use global default)
        operator_params = {}
        # Lookup order: exact op name → group key → global
        for op, group_key in [("qk", "qk"), ("av", "av"),
                               ("mlp_fc1", "mlp_fc1"), ("mlp_fc2", "mlp_fc2"),
                               ("input_proj", "input_proj"), ("proj", "proj")]:
            # Try exact op-level arg first, then group-level fallback
            a = getattr(args, f"mp_alpha_{op}", None)
            b = getattr(args, f"mp_beta_{op}", None)
            if a is None:
                # Group fallback: mlp_fc1/mlp_fc2 → mlp
                group_fallback = {"mlp_fc1": "mlp", "mlp_fc2": "mlp"}.get(op)
                if group_fallback:
                    a = getattr(args, f"mp_alpha_{group_fallback}", None)
            if b is None:
                group_fallback = {"mlp_fc1": "mlp", "mlp_fc2": "mlp"}.get(op)
                if group_fallback:
                    b = getattr(args, f"mp_beta_{group_fallback}", None)
            if a is not None or b is not None:
                operator_params[op] = (
                    a if a is not None else args.mp_alpha,
                    b if b is not None else args.mp_beta,
                )

        adaptive_config = AdaptiveMPConfig(
            stoc_len_levels=levels,
            alpha=args.mp_alpha,
            beta=args.mp_beta,
            enable_pruning=args.mp_enable_pruning,
            operator_params=operator_params,
            threshold_table_path=args.adaptive_mp_table,
        )
        sc_controller.init_adaptive_mp(adaptive_config)
        logging.info(f"Adaptive mixed precision V2 enabled: levels={levels}, "
                     f"alpha={args.mp_alpha}, beta={args.mp_beta}, "
                     f"pruning={args.mp_enable_pruning}, "
                     f"operator_params={operator_params}, "
                     f"threshold_table={args.adaptive_mp_table}")
    elif args.mp:
        levels = [int(x) for x in args.mp_levels.split(',')]
        fractions = ([float(x) for x in args.mp_fractions.split(',')]
                     if args.mp_fractions else None)
        mp_config = MPConfig(stoc_len_levels=levels, level_fractions=fractions)
        sc_controller.init_mp(mp_config)
        logging.info(f"Mixed precision enabled: levels={levels}, fractions={mp_config.level_fractions}")

    # Always enable metric profiling (lightweight, no overhead when MP is off)
    from qdit.sc_integration.mp_config import MetricProfiler
    MetricProfiler.enable()

    print("Finish quantization with SC support!")
    logging.info(model)

    # Generate sample images
    model.to(device)
    model.half()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    if args.generate_fid_samples and args.num_fid_samples > 0:
        # Skip validation and debug dumps — go straight to FID generation
        generate_samples_for_fid(args, model, diffusion, vae, sc_controller)
    else:
        validate_model_sc(args, model, diffusion, vae, sc_controller)

        # Dump SC vs FP comparison CSVs (all blocks, all timesteps)
        from qdit.sc_integration.sc_mlp import SCMlp
        from qdit.sc_integration.sc_attention import SCAttention
        from qdit.sc_integration.mp_config import MPDistributionLogger, MetricProfiler
        SCMlp.dump_compare_csv(f'{args.experiment_dir}/debug_sc_mlp.csv')
        SCAttention.dump_compare_csv(f'{args.experiment_dir}/debug_sc_proj.csv')
        MPDistributionLogger.summary(max_stoc_len=2 ** args.sc_prec,
                                       save_path=f'{args.experiment_dir}/mp_savings_summary.txt')
        MPDistributionLogger.dump_csv(f'{args.experiment_dir}/debug_mp_distribution.csv')
        MetricProfiler.dump_csv(f'{args.experiment_dir}/profile_metric_sigma.csv')

        if args.debug_sc:
            sc_controller.save_debug_log(f'{args.experiment_dir}/debug_sc.csv')

    # Final cleanup: release model and GPU memory
    del model, vae, diffusion
    torch.cuda.empty_cache()
    print("GPU memory released.")


def create_argparser():
    parser = argparse.ArgumentParser()

    # SC parameters
    parser.add_argument(
        '--timewise', type=float, default=0.0,
        help='Fraction of timesteps to use SC (0-1). SC used for first timewise*total_timesteps steps.'
    )
    parser.add_argument(
        '--qklayerwise', type=float, default=0.0,
        help='Fraction of blocks to use SC for q@k^T (0-1). SC used in first qklayerwise*total_blocks blocks.'
    )
    parser.add_argument(
        '--avlayerwise', type=float, default=0.0,
        help='Fraction of blocks to use SC for attn@v (0-1).'
    )
    parser.add_argument(
        '--projlayerwise', type=float, default=0.0,
        help='Fraction of blocks to use SC for output projection (0-1).'
    )
    parser.add_argument(
        '--mlplayerwise', type=float, default=0.0,
        help='Fraction of blocks to use SC for MLP fc1/fc2 (0-1).'
    )
    parser.add_argument(
        '--inputprojlayerwise', type=float, default=0.0,
        help='Fraction of blocks to use SC for QKV input projection (0-1).'
    )
    parser.add_argument(
        '--reverse_layerwise', action='store_true',
        help='Apply SC to the last N blocks instead of the first N blocks.'
    )
    parser.add_argument(
        '--sc_skip_blocks', type=str, default='',
        help='Comma-separated block indices to force FP (skip SC). E.g. "0,27".'
    )
    parser.add_argument(
        '--debug_sc', action='store_true',
        help='Debug mode: run both FP and SC paths, log per-operator error, use FP downstream.'
    )
    parser.add_argument(
        '--sc_prec', type=int, default=8,
        help='SC precision. Determines stoc_len=2^sc_prec and quant_max=2^(sc_prec-1)-1.'
    )
    parser.add_argument(
        '--sc_fixed_level_prec', action='store_true',
        help='Keep kernel sc_prec fixed to --sc_prec for all stoc_len levels. '
             'Without this flag, runtime derives sc_prec from each stoc_len via ceil(log2(stoc_len)).'
    )
    parser.add_argument(
        '--sc_halve', action='store_true',
        help='Halve the bipolar SC stream length (uSystolic / HUB sign-magnitude '
             'trick): run bipolar matmuls at stoc_len/2 with no accuracy loss. '
             'Only affects mode="bipolar" (i.e. --w_sym); ignored otherwise. '
             'Can also be enabled via the SC_HALVE=1 env var.'
    )
    parser.add_argument(
        '--sc_noise_model', action='store_true',
        help='Fast simulation: replace real SC kernels with closed-form '
             'analytical Gaussian noise surrogate (exact Bernoulli variance, '
             'no hyperparameters). Intended for '
             '50k FID sweeps where bitstream SC is too slow.'
    )
    parser.add_argument(
        '--sc_noise_local_correction', type=float, default=0.15,
        help='Variance correction for the noise surrogate when the matmul '
             'uses MULTIPLE local scales (per-row mlp / grouped, or per-batch '
             'batched_bipolar). Default 0.30 was fit against real SC kernels.'
    )
    parser.add_argument(
        '--sc_noise_global_correction', type=float, default=0.60,
        help='Variance correction for the noise surrogate when the matmul '
             'uses a SINGLE global scale (plain sc_matmul, per-tensor). '
             'Default 0.60 was fit against real SC kernels.'
    )
    parser.add_argument(
        '--av_attn_group_size', type=int, default=1,
        help='Row group size for attention in AV SC. 1=per-row (best), N=per-tensor (current).'
    )
    parser.add_argument(
        '--av_v_group_size', type=int, default=1,
        help='Row group size for V in AV SC. 1=per-feature (best), D=per-tensor (current).'
    )
    parser.add_argument(
        '--sc_qk_granularity', type=str, default='per_head',
        choices=['per_head', 'per_row'],
        help='QK SC quantization granularity. per_head (default, one scale per '
             'attention head) or per_row (per-query-row scaling, matches AV).'
    )
    parser.add_argument(
        '--sc_config', type=str, default=None,
        help='Path to JSON SC precision config. Overrides layerwise/sc_prec args with '
             'per-block, per-operator, per-group settings.'
    )
    parser.add_argument(
        '--save_sc_config', type=str, default=None,
        help='Save the effective SC config to JSON after setup (for inspection/reuse).'
    )

    # Mixed precision (per-token-row) parameters
    parser.add_argument(
        '--mp', action='store_true',
        help='Enable per-token-row mixed precision for QK/AV/MLP.'
    )
    parser.add_argument(
        '--mp_levels', type=str, default='256,128,64,32',
        help='Comma-separated stoc_len levels (sorted descending).'
    )
    parser.add_argument(
        '--mp_fractions', type=str, default=None,
        help='Comma-separated fractions per level (default: equal). E.g. "0.1,0.2,0.3,0.4".'
    )

    # Adaptive mixed precision (timestep-aware, inspired by APT)
    parser.add_argument(
        '--adaptive_mp', action='store_true',
        help='Enable adaptive mixed precision with timestep-aware thresholds.'
    )
    parser.add_argument(
        '--adaptive_mp_table', type=str, default=None,
        help='Path to a calibrated adaptive-MP threshold JSON table. '
             'When provided, runtime classification uses operator/timestep/layer '
             'bucket thresholds from the table and falls back to alpha/beta only '
             'for operators missing from the table.'
    )
    parser.add_argument(
        '--mp_alpha', type=float, default=0.3,
        help='Adaptive MP: threshold sensitivity to timestep progress.'
    )
    parser.add_argument(
        '--mp_beta', type=float, default=0.05,
        help='Adaptive MP: base threshold offset.'
    )
    parser.add_argument(
        '--mp_enable_pruning', action='store_true', default=True,
        help='Adaptive MP: enable row pruning (skip computation).'
    )
    parser.add_argument(
        '--no_mp_pruning', dest='mp_enable_pruning', action='store_false',
        help='Adaptive MP: disable row pruning.'
    )
    # Per-operator alpha/beta overrides
    parser.add_argument(
        '--mp_alpha_qk', type=float, default=None,
        help='Per-operator alpha for QK (default: use --mp_alpha).'
    )
    parser.add_argument(
        '--mp_alpha_av', type=float, default=None,
        help='Per-operator alpha for AV (default: use --mp_alpha).'
    )
    parser.add_argument(
        '--mp_alpha_mlp', type=float, default=None,
        help='Per-operator alpha for MLP fc1/fc2 (default: use --mp_alpha).'
    )
    parser.add_argument(
        '--mp_alpha_proj', type=float, default=None,
        help='Per-operator alpha for proj (default: use --mp_alpha).'
    )
    parser.add_argument(
        '--mp_alpha_input_proj', type=float, default=None,
        help='Per-operator alpha for input_proj (default: fallback to --mp_alpha_proj, then --mp_alpha).'
    )
    parser.add_argument(
        '--mp_alpha_mlp_fc1', type=float, default=None,
        help='Per-operator alpha for mlp_fc1 (default: fallback to --mp_alpha_mlp, then --mp_alpha).'
    )
    parser.add_argument(
        '--mp_alpha_mlp_fc2', type=float, default=None,
        help='Per-operator alpha for mlp_fc2 (default: fallback to --mp_alpha_mlp, then --mp_alpha).'
    )
    parser.add_argument(
        '--mp_beta_qk', type=float, default=None,
        help='Per-operator beta for QK (default: use --mp_beta).'
    )
    parser.add_argument(
        '--mp_beta_av', type=float, default=None,
        help='Per-operator beta for AV (default: use --mp_beta).'
    )
    parser.add_argument(
        '--mp_beta_mlp', type=float, default=None,
        help='Per-operator beta for MLP fc1/fc2 (default: use --mp_beta).'
    )
    parser.add_argument(
        '--mp_beta_proj', type=float, default=None,
        help='Per-operator beta for proj (default: use --mp_beta).'
    )
    parser.add_argument(
        '--mp_beta_input_proj', type=float, default=None,
        help='Per-operator beta for input_proj (default: fallback to --mp_beta_proj, then --mp_beta).'
    )
    parser.add_argument(
        '--mp_beta_mlp_fc1', type=float, default=None,
        help='Per-operator beta for mlp_fc1 (default: fallback to --mp_beta_mlp, then --mp_beta).'
    )
    parser.add_argument(
        '--mp_beta_mlp_fc2', type=float, default=None,
        help='Per-operator beta for mlp_fc2 (default: fallback to --mp_beta_mlp, then --mp_beta).'
    )

    # Range-based mixed precision (weight min/max range)
    parser.add_argument(
        '--range_mp', action='store_true',
        help='Enable range-based mixed precision using weight (max-min) range per group.'
    )
    parser.add_argument(
        '--range_mp_levels', type=str, default='256,128,64,32',
        help='Comma-separated stoc_len levels for range-based MP (sorted descending).'
    )
    parser.add_argument(
        '--range_mp_threshold', type=float, default=0.3,
        help='Range-based MP: normalized range threshold (0-1). Higher = more groups get lower precision.'
    )
    parser.add_argument(
        '--range_mp_threshold_qk', type=float, default=None,
        help='Per-operator range threshold for QK (default: use --range_mp_threshold).'
    )
    parser.add_argument(
        '--range_mp_threshold_av', type=float, default=None,
        help='Per-operator range threshold for AV (default: use --range_mp_threshold).'
    )
    parser.add_argument(
        '--range_mp_threshold_proj', type=float, default=None,
        help='Per-operator range threshold for proj (default: use --range_mp_threshold).'
    )
    parser.add_argument(
        '--range_mp_threshold_mlp', type=float, default=None,
        help='Per-operator range threshold for MLP fc1/fc2 (default: use --range_mp_threshold).'
    )
    parser.add_argument(
        '--range_mp_threshold_input_proj', type=float, default=None,
        help='Per-operator range threshold for input_proj (default: use --range_mp_threshold).'
    )

    # Quantization parameters (same as quant_main.py)
    parser.add_argument(
        '--wbits', type=int, default=8, choices=[2, 3, 4, 5, 6, 7, 8, 16],
        help='Bits for weight quantization (16 for no quantization).'
    )
    parser.add_argument(
        '--abits', type=int, default=8, choices=[2, 3, 4, 5, 6, 7, 8, 16],
        help='Bits for activation quantization (16 for no quantization).'
    )
    parser.add_argument(
        '--exponential', action='store_true',
        help='Use exponent-only for weight quantization.'
    )
    parser.add_argument(
        '--quantize_bmm_input', action='store_true',
        help='Quantize BMM input activations.'
    )
    parser.add_argument(
        '--a_sym', action='store_true',
        help='Use symmetric activation quantization.'
    )
    parser.add_argument(
        '--w_sym', action='store_true',
        help='Use symmetric weight quantization.'
    )
    parser.add_argument(
        '--static', action='store_true',
        help='Use static quantization for activations.'
    )
    parser.add_argument(
        '--weight_group_size', type=str, default='-1',
        help='Group size for weight quantization (-1 for per-tensor).'
    )
    parser.add_argument(
        '--weight_channel_group', type=int, default=1,
        help='Channel group size for weight quantization.'
    )
    parser.add_argument(
        '--act_group_size', type=str, default='-1',
        help='Group size for activation quantization (-1 for per-tensor).'
    )
    parser.add_argument(
        '--tiling', type=int, default=0, choices=[0, 16],
        help='Tile-wise quantization granularity.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of Hessian diagonal for dampening (GPTQ).'
    )
    parser.add_argument(
        '--use_gptq', action='store_true',
        help='Use GPTQ for weight quantization.'
    )
    parser.add_argument(
        '--quant_method', type=str, default='max', choices=['max', 'mse'],
        help='Weight quantization method.'
    )
    parser.add_argument(
        '--a_clip_ratio', type=float, default=1.0,
        help='Clip ratio for activation quantization.'
    )
    parser.add_argument(
        '--w_clip_ratio', type=float, default=1.0,
        help='Clip ratio for weight quantization.'
    )
    parser.add_argument(
        '--save_dir', type=str, default='../saved',
        help='Directory to save quantized weights.'
    )
    parser.add_argument(
        '--quant_type', type=str, default='int', choices=['int', 'fp'],
        help='Quantization data type.'
    )
    parser.add_argument(
        '--calib_data_path', type=str, default='../cali_data/cali_data_256.pth',
        help='Path to calibration data.'
    )

    # Model parameters (same as quant_main.py)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None, help="Path to DiT checkpoint.")
    parser.add_argument("--results-dir", type=str, default="../results")
    parser.add_argument("--save_ckpt", action="store_true", help="Save quantized checkpoint.")
    parser.add_argument("--tf32", action="store_true", help="Use TF32 matmuls.")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--generate-fid-samples", action="store_true", help="Generate samples for FID evaluation.")
    parser.add_argument(
        "--balanced_classes", action="store_true",
        help="Class-balanced FID sampling: produce contiguous blocks of "
             "(num_samples / num_classes) images per class id, so all images "
             "of the same class appear consecutively in the output."
    )
    parser.add_argument(
        "--class_start_idx", type=int, default=0,
        help="Starting index into the global balanced label list. Useful for "
             "splitting balanced sampling across multiple GPU processes "
             "(e.g. GPU 0: 0, GPU 1: num_samples_per_gpu, ...)."
    )
    parser.add_argument(
        "--balanced_total_samples", type=int, default=None,
        help="Total samples across ALL processes for class balance. "
             "If None, uses --num-fid-samples (single-GPU case). "
             "Multi-GPU: set this to the global FID target (e.g. 10000) "
             "so each GPU's class_start_idx slices into the correct global "
             "label list. Without this, multi-GPU runs duplicate classes."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="If an experiment_dir matching the current args already exists "
             "under --results-dir, reuse it (do not auto-increment to NNN+1) "
             "and continue FID sample generation from where it left off "
             "(skip batches whose samples/NNNNNN.png are already on disk). "
             "Class labels are deterministic from --class_start_idx so the "
             "resumed slice picks up exactly the missing labels. Re-generated "
             "noise z is NOT byte-identical to a fresh run (cuda_rng is reset), "
             "but the empirical distribution — and therefore FID — is unchanged."
    )
    parser.add_argument(
        "--target_indices_path", type=str, default=None,
        help="Path to a text file with one global sample index per line. "
             "When set, this overrides --class_start_idx and --num-fid-samples: "
             "the process will produce exactly the listed indices, writing each "
             "to {samples_dir}/{idx:06d}.png with the global index in the "
             "filename. Already-existing PNGs are skipped (idempotent resume). "
             "Requires --balanced_classes + --balanced_total_samples (so the "
             "label-by-index mapping is well-defined). Designed for sweep "
             "scripts to do work-stealing across an arbitrary GPU count."
    )
    parser.add_argument(
        "--samples_dir_override", type=str, default=None,
        help="Absolute path for sample PNG output, instead of "
             "{experiment_dir}/samples. Used by sweep scripts so multiple "
             "GPUs write to a single shared directory keyed by global index."
    )

    return parser


if __name__ == "__main__":
    main()
    if _shutdown_requested:
        # Non-zero exit so the sweep wrapper sees this GPU as FAILED and
        # skips the merge step (we don't want to merge a partial sample set
        # after a SIGINT/SIGTERM, e.g. spot preemption). Re-run with --resume
        # to continue from where this run left off.
        sys.exit(130)
