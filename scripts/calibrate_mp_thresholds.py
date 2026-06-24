#!/usr/bin/env python3
"""
Calibrate non-uniform mixed-precision thresholds from a quantized FP teacher.

This script runs a quantized Q-DiT model with SC wrappers in pure teacher mode
(all SC disabled), hooks into the wrapped attention / MLP blocks, and replays
selected operators at multiple stoc_len levels.  The resulting per-unit error
curves are used to fit budget-aware, bucketed thresholds:

    operator x timestep_bucket x layer_bucket -> [tau_0, tau_1, ...]

Unlike the existing adaptive MP path, this calibration does not assume a
single threshold with evenly spaced lower-precision boundaries.  Instead, it:

1. Measures real operator error against the quantized FP teacher.
2. Solves a per-bucket budget allocation over discrete stoc_len levels.
3. Converts the calibrated level counts into non-uniform thresholds over the
   normalized runtime importance metric.

The output JSON is intended to be consumed by a later runtime policy.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusion import create_diffusion
from models.models import DiT_models
from qdit.datautils import get_loader
from qdit.outlier import get_act_scales
from qdit.sc_integration import (
    OPERATORS,
    SCController,
    add_sc_wrapper,
    create_sc_controller_from_args,
    quantize_sc_model,
)
from scmp_kernels.sc import sc_matmul
from quant_sc_main import SCDiffusionWrapper, create_argparser
from utils.download import find_model


LINEAR_OPS = {"input_proj", "proj", "mlp_fc1", "mlp_fc2"}
DEFAULT_BUDGET_RATIO = 0.4


def _parse_csv_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _parse_csv_ints(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _expand_levels(value: str) -> list[int]:
    levels = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(levels) < 2:
        raise ValueError(f"Need at least 2 stoc_len levels, got {levels}")
    for i in range(len(levels) - 1):
        if levels[i] <= levels[i + 1]:
            raise ValueError(f"Levels must be strictly descending, got {levels}")
    return levels


def _select_timesteps(
    total_timesteps: int,
    explicit_timesteps: list[int],
    num_samples: int,
) -> list[int]:
    if explicit_timesteps:
        valid = sorted({t for t in explicit_timesteps if 0 <= t < total_timesteps}, reverse=True)
        if not valid:
            raise ValueError(
                f"No valid timesteps in {explicit_timesteps}; total_timesteps={total_timesteps}"
            )
        return valid

    if num_samples <= 0:
        return list(range(total_timesteps))[::-1]
    if num_samples >= total_timesteps:
        return list(range(total_timesteps))[::-1]

    grid = np.linspace(total_timesteps - 1, 0, num_samples)
    return sorted({int(round(x)) for x in grid}, reverse=True)


def _normalize_metric(metric: torch.Tensor) -> torch.Tensor:
    metric = metric.float()
    m_min = metric.min()
    m_max = metric.max()
    if (m_max - m_min).item() < 1e-8:
        return torch.ones_like(metric, dtype=torch.float32)
    return (metric - m_min) / (m_max - m_min)


def _relative_l2_rows(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_f = pred.float().reshape(pred.shape[0], -1)
    target_f = target.float().reshape(target.shape[0], -1)
    denom = target_f.norm(dim=-1).clamp_min(1e-8)
    return (pred_f - target_f).norm(dim=-1) / denom


def _relative_l2_heads(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # [B, H, N, N] -> [H, B*N*N]
    pred_h = pred.float().permute(1, 0, 2, 3).reshape(pred.shape[1], -1)
    target_h = target.float().permute(1, 0, 2, 3).reshape(target.shape[1], -1)
    denom = target_h.norm(dim=-1).clamp_min(1e-8)
    return (pred_h - target_h).norm(dim=-1) / denom


def _cosine_dist_rows(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cos_sim per row (lower = better, like L2)."""
    pf = pred.float().reshape(pred.shape[0], -1)
    tf = target.float().reshape(target.shape[0], -1)
    cos = torch.nn.functional.cosine_similarity(pf, tf, dim=-1, eps=1e-8)
    return 1.0 - cos


def _cosine_dist_heads(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ph = pred.float().permute(1, 0, 2, 3).reshape(pred.shape[1], -1)
    th = target.float().permute(1, 0, 2, 3).reshape(target.shape[1], -1)
    cos = torch.nn.functional.cosine_similarity(ph, th, dim=-1, eps=1e-8)
    return 1.0 - cos


# Globals dispatched from CLI flags --metric / --teacher.
_METRIC_ROWS = _relative_l2_rows
_METRIC_HEADS = _relative_l2_heads
_USE_FP_TEACHER = False


def _save_fp_weights(model) -> None:
    """Stash FP weights on each QLinearLayer before quantize_sc_model.

    Registered as buffers so model.to(device) carries them along.
    """
    for block in model.blocks:
        for ql in (block.attn.qkv, block.attn.proj,
                   block.mlp.fc1, block.mlp.fc2):
            ql.register_buffer('fp_weight', ql.weight.clone().detach())


def _fp_linear(qlayer, x: torch.Tensor) -> torch.Tensor:
    bias = qlayer.bias.float() if qlayer.bias is not None else None
    w = qlayer.fp_weight.float()
    if w.device != x.device:
        w = w.to(x.device)
        if bias is not None:
            bias = bias.to(x.device)
    return torch.nn.functional.linear(x.float(), w, bias)


def _with_bias_like(target: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    out = torch.zeros_like(target, dtype=torch.float32)
    if bias is not None:
        out = out + bias.float().view(*([1] * (target.dim() - 1)), -1)
    return out


def _resolve_level_sc_prec(module, stoc_len: int) -> int:
    return module.sc_controller.resolve_sc_prec(stoc_len)


def _resolve_level_rng_levels(module, stoc_len: int) -> int | None:
    return None


def _bucket_index(value: int, total: int, num_buckets: int) -> int:
    if num_buckets <= 1 or total <= 1:
        return 0
    ratio = value / max(total - 1, 1)
    return min(num_buckets - 1, int(ratio * num_buckets))


def _cost_assignments(
    errors: np.ndarray,
    costs: np.ndarray,
    budget_total: float,
) -> np.ndarray:
    n_units = errors.shape[0]
    min_cost = float(n_units * costs[-1])
    max_cost = float(n_units * costs[0])
    if budget_total <= min_cost:
        return np.full(n_units, len(costs) - 1, dtype=np.int64)
    if budget_total >= max_cost:
        return np.zeros(n_units, dtype=np.int64)

    def solve_for_lambda(lmbd: float) -> tuple[np.ndarray, float]:
        objective = errors + lmbd * costs[None, :]
        assignment = objective.argmin(axis=1)
        total_cost = float(costs[assignment].sum())
        return assignment, total_cost

    lo = 0.0
    hi = 1.0
    _, cost_hi = solve_for_lambda(hi)
    while cost_hi > budget_total and hi < 1e6:
        hi *= 2.0
        _, cost_hi = solve_for_lambda(hi)

    best = np.zeros(n_units, dtype=np.int64)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        cand, cost_mid = solve_for_lambda(mid)
        best = cand
        if cost_mid > budget_total:
            lo = mid
        else:
            hi = mid

    return best


def _thresholds_from_counts(metrics: np.ndarray, counts: np.ndarray) -> list[float]:
    if metrics.size == 0:
        return []
    sorted_metrics = np.sort(metrics)[::-1]
    thresholds: list[float] = []
    offset = 0
    for count in counts[:-1]:
        offset += int(count)
        if offset <= 0:
            thresholds.append(1.0)
        elif offset >= len(sorted_metrics):
            thresholds.append(0.0)
        else:
            thresholds.append(
                float(0.5 * (sorted_metrics[offset - 1] + sorted_metrics[offset]))
            )
    return thresholds


def _global_lambda(
    weighted_errors: list[np.ndarray],
    costs: np.ndarray,
    budget_ratio: float,
    ref: float,
    row_counts: list[float],
) -> float:
    """One shared Lagrange multiplier lambda across ALL groups for a GLOBAL,
    row-weighted budget -- the cross-layer allocation core (ported from
    CrucibleComputingGroup/scmp_llm PR #9). Budget flows across layers/operators
    instead of pinning each (op, t-bucket, l-bucket) to budget_ratio*ref.

    row_counts[g] = group g's TRUE (un-subsampled) per-forward row count R_g.
    rep_g = R_g/n_g prices each stored (subsampled) row as R_g/n_g real rows, so
    the per-row assignment and the R_g-weighted budget share one consistent
    objective -> uniform is feasible and the optimum is <= uniform. Without R_g
    weighting, attention (qk/av), with far more rows per forward than the linears,
    is under-counted and global overspends (realized avg_sl >> target)."""
    total_rows = float(sum(row_counts))
    if total_rows <= 0:
        return 0.0
    reps = [float(R) / max(int(e.shape[0]), 1)
            for e, R in zip(weighted_errors, row_counts)]
    global_budget = budget_ratio * ref * total_rows
    min_cost = float(total_rows * costs[-1])
    max_cost = float(total_rows * costs[0])
    if global_budget >= max_cost:
        return 0.0
    if global_budget <= min_cost:
        return 1e12

    def total_cost(lmbd: float) -> float:
        t = 0.0
        for e, R, rep in zip(weighted_errors, row_counts, reps):
            assign = (e + lmbd * rep * costs[None, :]).argmin(axis=1)
            t += float(R) * float(costs[assign].mean())
        return t

    lo, hi = 0.0, 1.0
    while total_cost(hi) > global_budget and hi < 1e12:
        hi *= 2.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if total_cost(mid) > global_budget:
            lo = mid
        else:
            hi = mid
    return hi


class ThresholdCalibrator:
    def __init__(
        self,
        levels: list[int],
        operators: Iterable[str],
        selected_timesteps: Iterable[int],
        total_timesteps: int,
        total_blocks: int,
        timestep_buckets: int,
        layer_buckets: int,
        budget_ratio: float,
        budget_ref_stoc_len: int | None,
        max_units_per_call: int,
        min_bucket_units: int,
        rng_seed: int,
        log_raw: bool = False,
        budget_scope: str = "per_bucket",
    ):
        self.levels = levels
        self.operators = set(operators)
        self.selected_timesteps = set(int(t) for t in selected_timesteps)
        self.total_timesteps = total_timesteps
        self.total_blocks = total_blocks
        self.timestep_buckets = timestep_buckets
        self.layer_buckets = layer_buckets
        self.budget_ratio = budget_ratio
        self.budget_ref_stoc_len = int(budget_ref_stoc_len) if budget_ref_stoc_len else max(levels)
        self.max_units_per_call = max_units_per_call
        self.min_bucket_units = min_bucket_units
        self.costs = np.asarray(levels, dtype=np.float64)
        self.records: dict[tuple[str, int, int], dict[str, list[np.ndarray]]] = defaultdict(
            lambda: {"metrics": [], "errors": []}
        )
        # Budget scope: "per_bucket" (each (op,t,l) pinned to budget_ratio*ref --
        # original) or "global" (one shared lambda over all groups; budget flows
        # across layers/operators -- PR#9 port). true_counts accumulates each
        # group's pre-subsample row count R_g, used for the R_g-weighted global lambda.
        self.budget_scope = budget_scope
        self.true_counts: dict[tuple[str, int, int], int] = defaultdict(int)
        self._rng = np.random.default_rng(rng_seed)
        # Optional full-resolution log keyed by raw (operator, block_idx, timestep) so the
        # bucketing can be re-derived offline (any timestep/layer_buckets) without re-probing.
        self.log_raw = log_raw
        self.raw_log: list[tuple[str, int, int, np.ndarray, np.ndarray]] = []

    def _use_operator(self, operator: str) -> bool:
        return operator in self.operators

    def _use_timestep(self, timestep: int) -> bool:
        return timestep in self.selected_timesteps

    def should_record(self, operator: str, block_idx: int, timestep: int) -> bool:
        if not self._use_operator(operator) or not self._use_timestep(timestep):
            return False
        if operator == "input_proj" and block_idx >= self.total_blocks - 1:
            return False
        if operator in {"mlp_fc1", "mlp_fc2"} and block_idx >= self.total_blocks - 2:
            return False
        return True

    def add(
        self,
        operator: str,
        block_idx: int,
        timestep: int,
        metric_norm: torch.Tensor,
        errors_by_level: list[torch.Tensor],
    ):
        if not self.should_record(operator, block_idx, timestep):
            return

        metrics = metric_norm.detach().float().reshape(-1).cpu().numpy()
        err = torch.stack([e.detach().float().reshape(-1) for e in errors_by_level], dim=-1)
        errors = err.cpu().numpy()

        if metrics.size == 0:
            return

        true_n = int(metrics.size)  # TRUE per-call row count (before subsampling)
        if self.max_units_per_call > 0 and metrics.size > self.max_units_per_call:
            keep = self._rng.choice(metrics.size, self.max_units_per_call, replace=False)
            metrics = metrics[keep]
            errors = errors[keep]

        if self.log_raw:
            self.raw_log.append((operator, int(block_idx), int(timestep), metrics, errors))

        t_bucket = _bucket_index(timestep, self.total_timesteps, self.timestep_buckets)
        l_bucket = _bucket_index(block_idx, self.total_blocks, self.layer_buckets)
        key = (operator, t_bucket, l_bucket)
        self.records[key]["metrics"].append(metrics)
        self.records[key]["errors"].append(errors)
        self.true_counts[key] += true_n

    def dump_raw(self, path: str):
        """Pickle the full-resolution per-(operator,block,timestep) records plus the
        metadata needed to re-fit any bucketing offline (see rebucket_from_raw.py)."""
        import pickle
        meta = {
            "levels": self.levels,
            "operators": sorted(self.operators),
            "total_timesteps": self.total_timesteps,
            "total_blocks": self.total_blocks,
            "budget_ratio": self.budget_ratio,
            "budget_ref_stoc_len": self.budget_ref_stoc_len,
            "min_bucket_units": self.min_bucket_units,
        }
        with open(path, "wb") as f:
            pickle.dump({"meta": meta, "raw_log": self.raw_log}, f, protocol=4)
        print(f"Wrote raw calibration log ({len(self.raw_log)} records) to {path}")

    def _rep_for_key(self, key) -> float:
        """rep_g = R_g / n_g for one group (cost scale for the global assignment)."""
        n = sum(int(m.size) for m in self.records[key]["metrics"])
        R = float(self.true_counts.get(key, n))
        return R / max(n, 1)

    def _rep_for_op(self, operator) -> float:
        n = R = 0
        for key, rec in self.records.items():
            if key[0] != operator:
                continue
            ng = sum(int(m.size) for m in rec["metrics"])
            n += ng
            R += self.true_counts.get(key, ng)
        return (R / max(n, 1)) if n else 1.0

    def _fit_group(self, metrics: np.ndarray, errors: np.ndarray,
                   lam: float | None = None, weight: float = 1.0,
                   rep: float = 1.0) -> dict:
        if lam is None:
            # Per-bucket budget: pin this group to budget_ratio*ref average.
            budget_total = self.budget_ratio * self.budget_ref_stoc_len * metrics.size
            assignment = _cost_assignments(errors, self.costs, budget_total)
        else:
            # Global cross-layer: assign at the shared price lambda, cost scaled by
            # rep_g = R_g/n_g so cheap (few-row) groups buy precision cheaply,
            # consistent with the R_g-weighted budget (see _global_lambda).
            objective = (weight * errors) + lam * rep * self.costs[None, :]
            assignment = objective.argmin(axis=1)
        counts = np.bincount(assignment, minlength=len(self.levels))
        thresholds = _thresholds_from_counts(metrics, counts)
        avg_cost = float(self.costs[assignment].mean()) if assignment.size else 0.0
        avg_error = float(errors[np.arange(errors.shape[0]), assignment].mean()) if assignment.size else 0.0
        level_mean_error = [float(errors[:, i].mean()) for i in range(errors.shape[1])]

        return {
            "num_units": int(metrics.size),
            "counts": counts.tolist(),
            "fractions": (counts / max(metrics.size, 1)).tolist(),
            "thresholds": thresholds,
            "avg_stoc_len": avg_cost,
            "avg_error": avg_error,
            "level_mean_error": level_mean_error,
            "metric_mean": float(metrics.mean()) if metrics.size else 0.0,
            "metric_std": float(metrics.std()) if metrics.size else 0.0,
        }

    def export(self) -> tuple[dict, list[dict]]:
        summary_rows: list[dict] = []
        payload = {
            "stoc_len_levels": self.levels,
            "budget_ratio": self.budget_ratio,
            "budget_ref_stoc_len": self.budget_ref_stoc_len,
            "timestep_buckets": self.timestep_buckets,
            "layer_buckets": self.layer_buckets,
            "budget_scope": self.budget_scope,
            "operator_defaults": {},
            "buckets": {},
        }

        # Global cross-layer: ONE shared lambda over all per-bucket groups so the
        # budget flows across layers/operators (uniform importance W_g=1; this is
        # PR#9's `act_global` signal). per_bucket leaves lam=None (each group
        # independently pinned to budget_ratio*ref -- original behaviour).
        global_lam = None
        if self.budget_scope == "global":
            werrs, row_counts = [], []
            for key, rec in self.records.items():
                e = np.concatenate(rec["errors"], axis=0)
                werrs.append(e)
                row_counts.append(float(self.true_counts.get(key, e.shape[0])))
            global_lam = _global_lambda(
                werrs, self.costs, self.budget_ratio,
                float(self.budget_ref_stoc_len), row_counts)
            payload["global_lambda"] = float(global_lam)

        for operator in sorted(self.operators):
            op_metrics = []
            op_errors = []
            for (op, _, _), rec in self.records.items():
                if op != operator:
                    continue
                op_metrics.extend(rec["metrics"])
                op_errors.extend(rec["errors"])
            if op_metrics:
                fitted = self._fit_group(
                    np.concatenate(op_metrics, axis=0),
                    np.concatenate(op_errors, axis=0),
                    lam=global_lam, weight=1.0, rep=self._rep_for_op(operator),
                )
                payload["operator_defaults"][operator] = fitted
                summary_rows.append(
                    {
                        "scope": "operator_default",
                        "operator": operator,
                        "t_bucket": -1,
                        "l_bucket": -1,
                        **_flatten_summary(fitted),
                    }
                )

        for key in sorted(self.records.keys()):
            metrics_list = self.records[key]["metrics"]
            errors_list = self.records[key]["errors"]
            metrics = np.concatenate(metrics_list, axis=0)
            errors = np.concatenate(errors_list, axis=0)
            if metrics.size < self.min_bucket_units:
                continue
            operator, t_bucket, l_bucket = key
            fitted = self._fit_group(metrics, errors, lam=global_lam,
                                     weight=1.0, rep=self._rep_for_key(key))
            bucket_key = f"{operator}:t{t_bucket}:l{l_bucket}"
            payload["buckets"][bucket_key] = fitted
            summary_rows.append(
                {
                    "scope": "bucket",
                    "operator": operator,
                    "t_bucket": t_bucket,
                    "l_bucket": l_bucket,
                    **_flatten_summary(fitted),
                }
            )

        # Honest iso-budget check: row-weighted (R_g) average stoc_len across all
        # buckets. For per_bucket this is ~budget_ratio*ref by construction; for
        # global it's the realized average -- should also land near the target.
        num = den = 0.0
        for bkey, fitted in payload["buckets"].items():
            op, tpart, lpart = bkey.split(":")
            k = (op, int(tpart[1:]), int(lpart[1:]))
            R = float(self.true_counts.get(k, fitted["num_units"]))
            num += R * fitted["avg_stoc_len"]
            den += R
        payload["expected_avg_stoc_len"] = (num / den) if den else 0.0

        return payload, summary_rows


def _flatten_summary(fitted: dict) -> dict[str, str | int | float]:
    row = {
        "num_units": fitted["num_units"],
        "avg_stoc_len": fitted["avg_stoc_len"],
        "avg_error": fitted["avg_error"],
        "metric_mean": fitted["metric_mean"],
        "metric_std": fitted["metric_std"],
    }
    row["counts"] = ",".join(str(int(x)) for x in fitted["counts"])
    row["fractions"] = ",".join(f"{float(x):.6f}" for x in fitted["fractions"])
    row["thresholds"] = ",".join(f"{float(x):.6f}" for x in fitted["thresholds"])
    row["level_mean_error"] = ",".join(f"{float(x):.6f}" for x in fitted["level_mean_error"])
    return row


def _linear_teacher_pruned(target: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    return _with_bias_like(target, bias)


def _run_attention_linear_level(
    module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stoc_len: int,
    chunk_d: int,
    grouped: bool,
) -> torch.Tensor:
    if stoc_len == 0:
        out_features = weight.shape[0]
        target = torch.empty(*x.shape[:-1], out_features, device=x.device, dtype=torch.float32)
        return _linear_teacher_pruned(target, bias)
    sc_prec = _resolve_level_sc_prec(module, stoc_len)
    rng_levels = _resolve_level_rng_levels(module, stoc_len)
    if grouped:
        orig_shape = x.shape
        d_model = x.shape[-1]
        x_flat = x.reshape(-1, d_model)

        if chunk_d > 0 and d_model > chunk_d:
            result = None
            for start in range(0, d_model, chunk_d):
                end = min(start + chunk_d, d_model)
                x_chunk = x_flat[:, start:end].contiguous()
                w_chunk = weight[:, start:end].contiguous()
                config = module._get_sc_config(end - start, sc_prec)
                chunk_result = sc_matmul(
                    x_chunk,
                    w_chunk,
                    granularity="per_row",
                    mode=module.sc_mode,
                    sc_prec=sc_prec,
                    config=config,
                    group_a=1,
                    group_b=1,
                    chunk_d=chunk_d,
                    stoc_len=stoc_len,
                    rng_levels=rng_levels,
                )
                result = chunk_result if result is None else result + chunk_result
        else:
            config = module._get_sc_config(d_model, sc_prec)
            result = sc_matmul(
                    x_flat,
                    weight,
                    granularity="per_row",
                    mode=module.sc_mode,
                    sc_prec=sc_prec,
                    config=config,
                    group_a=1,
                    group_b=1,
                    stoc_len=stoc_len,
                    rng_levels=rng_levels,
                )

        if bias is not None:
            result = result + bias
        return result.reshape(*orig_shape[:-1], -1).float()

    return module._sc_linear_uniform(
        x,
        weight,
        bias,
        sc_prec,
        stoc_len,
        chunk_d=chunk_d,
        grouped=grouped,
    ).float()


def _run_mlp_linear_level(
    module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stoc_len: int,
    chunk_d: int,
) -> torch.Tensor:
    if stoc_len == 0:
        out_features = weight.shape[0]
        target = torch.empty(*x.shape[:-1], out_features, device=x.device, dtype=torch.float32)
        return _linear_teacher_pruned(target, bias)
    sc_prec = _resolve_level_sc_prec(module, stoc_len)
    rng_levels = _resolve_level_rng_levels(module, stoc_len)
    orig_shape = x.shape
    d_model = x.shape[-1]
    x_flat = x.reshape(-1, d_model)

    if chunk_d > 0 and d_model > chunk_d:
        result = None
        for start in range(0, d_model, chunk_d):
            end = min(start + chunk_d, d_model)
            x_chunk = x_flat[:, start:end].contiguous()
            w_chunk = weight[:, start:end].contiguous()
            config = module._get_sc_config(end - start, sc_prec)
            chunk_result = sc_matmul(
                x_chunk,
                w_chunk,
                granularity="per_row",
                mode=module.sc_mode,
                sc_prec=sc_prec,
                config=config,
                group_a=1,
                group_b=1,
                chunk_d=chunk_d,
                stoc_len=stoc_len,
                rng_levels=rng_levels,
            )
            result = chunk_result if result is None else result + chunk_result
    else:
        config = module._get_sc_config(d_model, sc_prec)
        result = sc_matmul(
            x_flat,
            weight,
            granularity="per_row",
            mode=module.sc_mode,
            sc_prec=sc_prec,
            config=config,
            group_a=1,
            group_b=1,
            chunk_d=chunk_d,
            stoc_len=stoc_len,
            rng_levels=rng_levels,
        )
    if bias is not None:
        result = result + bias
    return result.reshape(*orig_shape[:-1], -1).float()


def _build_input_proj_output(module, x_q: torch.Tensor, stoc_len: int) -> torch.Tensor:
    w = module.qkv.weight
    b = module.qkv.bias
    c = w.shape[0] // 3
    q = _run_attention_linear_level(module, x_q, w[:c], b[:c] if b is not None else None, stoc_len, 144, True)
    k = _run_attention_linear_level(module, x_q, w[c:2 * c], b[c:2 * c] if b is not None else None, stoc_len, 144, True)
    v = _run_attention_linear_level(module, x_q, w[2 * c:], b[2 * c:] if b is not None else None, stoc_len, 144, True)
    return torch.cat([q, k, v], dim=-1)


def _write_summary_csv(path: str, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class CalibrationRunner:
    def __init__(self, args, model, diffusion, controller, calibrator: ThresholdCalibrator):
        self.args = args
        self.model = model
        self.diffusion = diffusion
        self.controller = controller
        self.calibrator = calibrator

    def _attn_hook(self, module, inputs, _output):
        timestep = module.sc_controller.current_timestep
        if timestep is None or timestep not in self.calibrator.selected_timesteps:
            return

        x = inputs[0]
        block_idx = module.block_idx

        with torch.no_grad(), autocast(enabled=False):
            if module.reorder_index_qkv is not None:
                x_q = torch.index_select(x, 2, module.reorder_index_qkv)
            else:
                x_q = x
            x_q = module.input_quant(x_q)
            x_q = x_q.to(dtype=module.qkv.weight.dtype)

            # input projection
            if self.calibrator.should_record("input_proj", block_idx, timestep):
                if _USE_FP_TEACHER:
                    x_for_teacher = (torch.index_select(x, 2, module.reorder_index_qkv)
                                     if module.reorder_index_qkv is not None else x)
                    teacher_qkv = _fp_linear(module.qkv, x_for_teacher).to(dtype=x_q.dtype)
                else:
                    teacher_qkv = module.qkv(x_q)
                row_metric = _normalize_metric(x_q.reshape(-1, x_q.shape[-1]).abs().amax(dim=-1))
                level_errors = []
                for sl in self.calibrator.levels:
                    sc_qkv = _build_input_proj_output(module, x_q, sl)
                    level_errors.append(
                        _METRIC_ROWS(
                            sc_qkv.reshape(-1, sc_qkv.shape[-1]),
                            teacher_qkv.float().reshape(-1, teacher_qkv.shape[-1]),
                        )
                    )
                self.calibrator.add("input_proj", block_idx, timestep, row_metric, level_errors)
            else:
                if _USE_FP_TEACHER:
                    x_for_teacher = (torch.index_select(x, 2, module.reorder_index_qkv)
                                     if module.reorder_index_qkv is not None else x)
                    teacher_qkv = _fp_linear(module.qkv, x_for_teacher).to(dtype=x_q.dtype)
                else:
                    teacher_qkv = module.qkv(x_q)

            bsz, n_tokens, _ = teacher_qkv.shape
            qkv = teacher_qkv.reshape(
                bsz, n_tokens, 3, module.num_heads, module.head_dim
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q = module.q_norm(q)
            k = module.k_norm(k)

            if module.quantize_bmm_input:
                q = module.q_quant(q)
                k = module.k_quant(k)
                v = module.v_quant(v)

            q_scaled = q * module.scale
            teacher_attn = (q_scaled @ k.transpose(-2, -1))

            if self.calibrator.should_record("qk", block_idx, timestep):
                q_metric = _normalize_metric(q_scaled.float().abs().amax(dim=(0, 2, 3)))
                level_errors = []
                for sl in self.calibrator.levels:
                    if sl == 0:
                        sc_attn = torch.zeros_like(teacher_attn, dtype=torch.float32)
                    else:
                        sc_prec = _resolve_level_sc_prec(module, sl)
                        sc_attn = module._sc_qk_uniform(q, k, sc_prec, sl).float()
                    level_errors.append(_METRIC_HEADS(sc_attn, teacher_attn.float()))
                self.calibrator.add("qk", block_idx, timestep, q_metric, level_errors)

            attn = module.attn_drop(teacher_attn.softmax(dim=-1))
            teacher_av = (attn @ v)

            if self.calibrator.should_record("av", block_idx, timestep):
                attn_flat = attn.reshape(-1, attn.shape[-2], attn.shape[-1])
                teacher_flat = teacher_av.reshape(-1, teacher_av.shape[-2], teacher_av.shape[-1])
                level_errors_by_level: list[list[torch.Tensor]] = [[] for _ in self.calibrator.levels]
                for sl in self.calibrator.levels:
                    if sl == 0:
                        sc_av = torch.zeros_like(teacher_av, dtype=torch.float32)
                    else:
                        sc_prec = _resolve_level_sc_prec(module, sl)
                        sc_av = module._sc_av_uniform(attn, v, sc_prec, sl).float()
                    sc_flat = sc_av.reshape(-1, sc_av.shape[-2], sc_av.shape[-1])
                    for i in range(sc_flat.shape[0]):
                        level_errors_by_level[self.calibrator.levels.index(sl)].append(
                            _METRIC_ROWS(sc_flat[i], teacher_flat[i].float())
                        )

                for i in range(attn_flat.shape[0]):
                    row_metric = _normalize_metric(attn_flat[i].amax(dim=-1))
                    level_errors = [
                        level_errors_by_level[level_idx][i]
                        for level_idx in range(len(self.calibrator.levels))
                    ]
                    self.calibrator.add("av", block_idx, timestep, row_metric, level_errors)

            proj_hidden = module.num_heads * module.head_dim
            proj_in = teacher_av.transpose(1, 2).reshape(bsz, n_tokens, proj_hidden)
            if module.reorder_index_proj is not None:
                proj_in = torch.index_select(proj_in, 2, module.reorder_index_proj)
            proj_in = module.act_quant(proj_in)
            proj_in = proj_in.to(dtype=module.proj.weight.dtype)

            if self.calibrator.should_record("proj", block_idx, timestep):
                if _USE_FP_TEACHER:
                    proj_in_for_teacher = teacher_av.transpose(1, 2).reshape(bsz, n_tokens, proj_hidden)
                    if module.reorder_index_proj is not None:
                        proj_in_for_teacher = torch.index_select(proj_in_for_teacher, 2,
                                                                  module.reorder_index_proj)
                    teacher_proj = _fp_linear(module.proj, proj_in_for_teacher).to(dtype=proj_in.dtype)
                else:
                    teacher_proj = module.proj(proj_in)
                row_metric = _normalize_metric(proj_in.reshape(-1, proj_in.shape[-1]).abs().amax(dim=-1))
                level_errors = []
                for sl in self.calibrator.levels:
                    sc_proj = _run_attention_linear_level(
                        module, proj_in, module.proj.weight, module.proj.bias, sl, 144, True
                    )
                    level_errors.append(
                        _METRIC_ROWS(
                            sc_proj.reshape(-1, sc_proj.shape[-1]),
                            teacher_proj.float().reshape(-1, teacher_proj.shape[-1]),
                        )
                    )
                self.calibrator.add("proj", block_idx, timestep, row_metric, level_errors)

    def _mlp_hook(self, module, inputs, _output):
        timestep = module.sc_controller.current_timestep
        if timestep is None or timestep not in self.calibrator.selected_timesteps:
            return

        block_idx = module.block_idx
        x = inputs[0]

        with torch.no_grad(), autocast(enabled=False):
            if module.reorder_index_fc1 is not None:
                x_fc1 = torch.index_select(x, 2, module.reorder_index_fc1)
            else:
                x_fc1 = x
            x_fc1 = module.input_quant(x_fc1)
            x_fc1 = x_fc1.to(dtype=module.fc1.weight.dtype)

            if self.calibrator.should_record("mlp_fc1", block_idx, timestep):
                if _USE_FP_TEACHER:
                    x_for_teacher = (torch.index_select(x, 2, module.reorder_index_fc1)
                                     if module.reorder_index_fc1 is not None else x)
                    teacher_fc1 = _fp_linear(module.fc1, x_for_teacher).to(dtype=x_fc1.dtype)
                else:
                    teacher_fc1 = module.fc1(x_fc1)
                row_metric = _normalize_metric(x_fc1.reshape(-1, x_fc1.shape[-1]).abs().amax(dim=-1))
                level_errors = []
                for sl in self.calibrator.levels:
                    sc_fc1 = _run_mlp_linear_level(
                        module, x_fc1, module.fc1.weight, module.fc1.bias, sl, 72
                    )
                    level_errors.append(
                        _METRIC_ROWS(
                            sc_fc1.reshape(-1, sc_fc1.shape[-1]),
                            teacher_fc1.float().reshape(-1, teacher_fc1.shape[-1]),
                        )
                    )
                self.calibrator.add("mlp_fc1", block_idx, timestep, row_metric, level_errors)
            else:
                if _USE_FP_TEACHER:
                    x_for_teacher = (torch.index_select(x, 2, module.reorder_index_fc1)
                                     if module.reorder_index_fc1 is not None else x)
                    teacher_fc1 = _fp_linear(module.fc1, x_for_teacher).to(dtype=x_fc1.dtype)
                else:
                    teacher_fc1 = module.fc1(x_fc1)

            x_mid = module.act(teacher_fc1)
            x_mid = module.drop1(x_mid)
            x_mid = module.norm(x_mid)
            x_mid = module.act_quant(x_mid)
            x_mid = x_mid.to(dtype=module.fc2.weight.dtype)

            if self.calibrator.should_record("mlp_fc2", block_idx, timestep):
                if _USE_FP_TEACHER:
                    teacher_fc2 = _fp_linear(module.fc2, x_mid).to(dtype=x_mid.dtype)
                else:
                    teacher_fc2 = module.fc2(x_mid)
                row_metric = _normalize_metric(x_mid.reshape(-1, x_mid.shape[-1]).abs().amax(dim=-1))
                level_errors = []
                for sl in self.calibrator.levels:
                    sc_fc2 = _run_mlp_linear_level(
                        module, x_mid, module.fc2.weight, module.fc2.bias, sl, 72
                    )
                    level_errors.append(
                        _METRIC_ROWS(
                            sc_fc2.reshape(-1, sc_fc2.shape[-1]),
                            teacher_fc2.float().reshape(-1, teacher_fc2.shape[-1]),
                        )
                    )
                self.calibrator.add("mlp_fc2", block_idx, timestep, row_metric, level_errors)

    def register_hooks(self):
        hooks = []
        for block in self.model.blocks:
            hooks.append(block.attn.register_forward_hook(self._attn_hook))
            hooks.append(block.mlp.register_forward_hook(self._mlp_hook))
        return hooks


def _disable_sc(controller):
    for block_idx in range(controller.total_blocks):
        for op in OPERATORS:
            controller.precision_map.set(block_idx, op, enabled=False, timewise=0.0)
    controller.mp_config = None
    controller.adaptive_mp_config = None
    controller.range_mp_config = None


def _build_parser():
    parser = create_argparser()
    parser.description = "Calibrate bucketed non-uniform MP thresholds from a quantized FP teacher."
    parser.add_argument(
        "--calib_output_json",
        type=str,
        default="threshold_mp_calibration.json",
        help="Path to write the calibrated threshold table JSON.",
    )
    parser.add_argument(
        "--calib_summary_csv",
        type=str,
        default="threshold_mp_calibration_summary.csv",
        help="Path to write the per-bucket summary CSV.",
    )
    parser.add_argument(
        "--operators",
        type=str,
        default="input_proj,proj,mlp_fc1,mlp_fc2,qk,av",
        help="Comma-separated operators to calibrate.",
    )
    parser.add_argument(
        "--calib_timesteps",
        type=str,
        default=None,
        help="Optional comma-separated timestep list. Default: evenly spaced samples.",
    )
    parser.add_argument(
        "--dump_raw",
        type=str,
        default=None,
        help="If set, pickle the full-resolution per-(operator,block,timestep) error log "
             "to this path so any bucketing can be re-derived offline (rebucket_from_raw.py).",
    )
    parser.add_argument(
        "--budget_scope",
        type=str,
        choices=["per_bucket", "global"],
        default="per_bucket",
        help="Budget allocation scope. 'per_bucket' (default, original): each "
             "(op,t-bucket,l-bucket) pinned to budget_ratio*ref average. 'global' "
             "(PR#9 port): one shared lambda over all groups so budget flows across "
             "layers/operators (R_g-weighted -> optimum provably <= uniform).",
    )
    parser.add_argument(
        "--num_calib_timesteps",
        type=int,
        default=6,
        help="Number of timesteps to calibrate when --calib_timesteps is not provided.",
    )
    parser.add_argument(
        "--num_calib_batches",
        type=int,
        default=1,
        help="Number of random diffusion trajectories to run for calibration.",
    )
    parser.add_argument(
        "--timestep_buckets",
        type=int,
        default=4,
        help="Number of timestep buckets in the output threshold table.",
    )
    parser.add_argument(
        "--layer_buckets",
        type=int,
        default=4,
        help="Number of layer buckets in the output threshold table.",
    )
    parser.add_argument(
        "--budget_ratio",
        type=float,
        default=DEFAULT_BUDGET_RATIO,
        help="Target average stoc budget relative to uniform max stoc_len. "
             f"Default: {DEFAULT_BUDGET_RATIO}.",
    )
    parser.add_argument(
        "--budget_ref_stoc_len",
        type=int,
        default=None,
        help="Reference stoc_len used to interpret --budget_ratio. "
             "Default: max(mp_levels). Set to 256 to express budget against a 256 baseline.",
    )
    parser.add_argument(
        "--max_units_per_call",
        type=int,
        default=0,
        help="Randomly subsample units per operator call to bound runtime/memory. "
             "0 (default) = NO subsampling -> every operator recorded full per-row, so "
             "stored n_g == true rows R_g and rep_g==1 uniformly. This removes the "
             "subsampling asymmetry (av uncapped vs linears capped) that distorted the "
             "global row-weighted allocation.",
    )
    parser.add_argument(
        "--min_bucket_units",
        type=int,
        default=64,
        help="Minimum units required before emitting a bucket-specific threshold entry.",
    )
    parser.add_argument(
        "--teacher_cfg_scale",
        type=float,
        default=0.0,
        help="CFG scale to use during calibration trajectories. Default: 0 for speed.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["l2", "cosine"],
        default="cosine",
        help="Per-unit error metric. cosine = 1 - cos_sim (recommended for fix mode).",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        choices=["quant", "fp"],
        default="fp",
        help="Reference for measuring SC error. 'fp' uses FP weights × raw activation; "
             "'quant' uses w8a8 weights × quantized activation (legacy, degenerate in fix mode).",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Calibration currently requires CUDA / Triton kernels.")

    seed_everything(args.seed)
    device = "cuda"

    levels = _expand_levels(args.mp_levels)
    budget_ratio = args.budget_ratio
    operators = _parse_csv_set(args.operators)
    explicit_timesteps = _parse_csv_ints(args.calib_timesteps)
    budget_ref_stoc_len = args.budget_ref_stoc_len or max(levels)

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))

    args.weight_group_size = eval(args.weight_group_size)
    args.act_group_size = eval(args.act_group_size)
    if isinstance(args.weight_group_size, int):
        args.weight_group_size = [args.weight_group_size] * len(model.blocks)
    if isinstance(args.act_group_size, int):
        args.act_group_size = [args.act_group_size] * len(model.blocks)

    sc_controller = create_sc_controller_from_args(args, model)
    _disable_sc(sc_controller)

    if args.static:
        dataloader = get_loader(args.calib_data_path, nsamples=1024, batch_size=16)
        scales = get_act_scales(model, diffusion, dataloader, device, args)
    else:
        scales = defaultdict(lambda: None)

    print("Adding SC wrappers for calibration...")
    model = add_sc_wrapper(model, device, args, scales, sc_controller)
    if args.teacher == "fp":
        print("Stashing FP weights for FP teacher...")
        _save_fp_weights(model)
    print("Quantizing wrapped model...")
    model = quantize_sc_model(model, device, args, sc_controller=sc_controller)

    # Wire metric + teacher dispatch globals.
    global _METRIC_ROWS, _METRIC_HEADS, _USE_FP_TEACHER
    if args.metric == "cosine":
        _METRIC_ROWS = _cosine_dist_rows
        _METRIC_HEADS = _cosine_dist_heads
    _USE_FP_TEACHER = (args.teacher == "fp")
    print(f"Calibration metric: {args.metric}, teacher: {args.teacher}")
    model.to(device)
    model.eval().half()

    selected_timesteps = _select_timesteps(
        diffusion.num_timesteps,
        explicit_timesteps,
        args.num_calib_timesteps,
    )
    print(f"Selected timesteps for calibration: {selected_timesteps}")
    print(f"Operators: {sorted(operators)}")
    print(
        f"Levels: {levels}, budget_ratio={budget_ratio:.4f}, "
        f"budget_ref_stoc_len={budget_ref_stoc_len}"
    )

    calibrator = ThresholdCalibrator(
        levels=levels,
        operators=operators,
        selected_timesteps=selected_timesteps,
        total_timesteps=diffusion.num_timesteps,
        total_blocks=len(model.blocks),
        timestep_buckets=args.timestep_buckets,
        layer_buckets=args.layer_buckets,
        budget_ratio=budget_ratio,
        budget_ref_stoc_len=budget_ref_stoc_len,
        max_units_per_call=args.max_units_per_call,
        min_bucket_units=args.min_bucket_units,
        rng_seed=args.seed,
        log_raw=bool(args.dump_raw),
        budget_scope=args.budget_scope,
    )
    runner = CalibrationRunner(args, model, diffusion, sc_controller, calibrator)
    hooks = runner.register_hooks()

    sc_diffusion = SCDiffusionWrapper(diffusion, sc_controller)
    cuda_rng = torch.Generator(device="cuda").manual_seed(args.seed)

    try:
        with torch.no_grad(), autocast():
            for batch_idx in range(args.num_calib_batches):
                n = args.batch_size
                z = torch.randn(
                    n, 4, latent_size, latent_size,
                    device=device, generator=cuda_rng,
                ).half()
                y = torch.randint(
                    0, args.num_classes, (n,),
                    device=device, generator=cuda_rng,
                )

                if args.teacher_cfg_scale and args.teacher_cfg_scale > 0:
                    z = torch.cat([z, z], 0)
                    y_null = torch.tensor([1000] * n, device=device)
                    y = torch.cat([y, y_null], 0)
                    model_kwargs = dict(y=y, cfg_scale=args.teacher_cfg_scale)
                else:
                    model_kwargs = dict(y=y)

                print(f"[Calibration] batch {batch_idx + 1}/{args.num_calib_batches}")
                _ = sc_diffusion.ddim_sample_loop(
                    model,
                    z.shape,
                    z,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=True,
                    device=device,
                )
    finally:
        for hook in hooks:
            hook.remove()

    payload, summary_rows = calibrator.export()
    payload["selected_timesteps"] = selected_timesteps
    payload["operators"] = sorted(operators)
    print(f"[calib] budget_scope={payload['budget_scope']} "
          f"global_lambda={payload.get('global_lambda', 'n/a')} "
          f"expected_avg_stoc_len={payload.get('expected_avg_stoc_len', 0):.2f} "
          f"(target={budget_ratio * budget_ref_stoc_len:.1f})")

    output_json_path = Path(args.calib_output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path = Path(args.calib_summary_csv)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w") as f:
        json.dump(payload, f, indent=2)
    _write_summary_csv(str(summary_csv_path), summary_rows)

    print(f"Wrote calibration JSON to {output_json_path}")
    print(f"Wrote summary CSV to {summary_csv_path}")

    if args.dump_raw:
        Path(args.dump_raw).parent.mkdir(parents=True, exist_ok=True)
        calibrator.dump_raw(args.dump_raw)


if __name__ == "__main__":
    main()
