#!/usr/bin/env python3
"""Re-derive an MP calibration table at ANY (timestep_buckets, layer_buckets) from a
raw calibration log dumped by calibrate_mp_thresholds.py --dump_raw, with NO GPU and
NO re-probing. The fitting logic is copied verbatim from calibrate_mp_thresholds.py so
the output JSON is bit-for-bit what a fresh probe at that bucketing would have produced.

Usage:
    python rebucket_from_raw.py <raw.pkl> <out.json> [TIMESTEP_BUCKETS] [LAYER_BUCKETS]
Defaults: TIMESTEP_BUCKETS=4, LAYER_BUCKETS=4.
"""
import json
import pickle
import sys
from collections import defaultdict

import numpy as np


# --- verbatim from calibrate_mp_thresholds.py ---
def _bucket_index(value: int, total: int, num_buckets: int) -> int:
    if num_buckets <= 1 or total <= 1:
        return 0
    ratio = value / max(total - 1, 1)
    return min(num_buckets - 1, int(ratio * num_buckets))


def _cost_assignments(errors, costs, budget_total):
    n_units = errors.shape[0]
    min_cost = float(n_units * costs[-1])
    max_cost = float(n_units * costs[0])
    if budget_total <= min_cost:
        return np.full(n_units, len(costs) - 1, dtype=np.int64)
    if budget_total >= max_cost:
        return np.zeros(n_units, dtype=np.int64)

    def solve_for_lambda(lmbd):
        objective = errors + lmbd * costs[None, :]
        assignment = objective.argmin(axis=1)
        return assignment, float(costs[assignment].sum())

    lo, hi = 0.0, 1.0
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


def _thresholds_from_counts(metrics, counts):
    if metrics.size == 0:
        return []
    sorted_metrics = np.sort(metrics)[::-1]
    thresholds, offset = [], 0
    for count in counts[:-1]:
        offset += int(count)
        if offset <= 0:
            thresholds.append(1.0)
        elif offset >= len(sorted_metrics):
            thresholds.append(0.0)
        else:
            thresholds.append(float(0.5 * (sorted_metrics[offset - 1] + sorted_metrics[offset])))
    return thresholds


def _fit_group(metrics, errors, costs, levels, budget_ratio, budget_ref_stoc_len):
    budget_total = budget_ratio * budget_ref_stoc_len * metrics.size
    assignment = _cost_assignments(errors, costs, budget_total)
    counts = np.bincount(assignment, minlength=len(levels))
    thresholds = _thresholds_from_counts(metrics, counts)
    avg_cost = float(costs[assignment].mean()) if assignment.size else 0.0
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


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    raw_path, out_json = sys.argv[1], sys.argv[2]
    TB = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    LB = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    with open(raw_path, "rb") as f:
        blob = pickle.load(f)
    meta, raw_log = blob["meta"], blob["raw_log"]
    levels = meta["levels"]
    costs = np.asarray(levels, dtype=np.float64)
    total_t, total_b = meta["total_timesteps"], meta["total_blocks"]
    br, bref = meta["budget_ratio"], meta["budget_ref_stoc_len"]
    min_bucket_units = meta["min_bucket_units"]
    operators = meta["operators"]

    buckets = defaultdict(lambda: {"m": [], "e": []})
    op_all = defaultdict(lambda: {"m": [], "e": []})
    ts_seen = set()
    for op, block, t, m, e in raw_log:
        ts_seen.add(int(t))
        tb = _bucket_index(t, total_t, TB)
        lb = _bucket_index(block, total_b, LB)
        buckets[(op, tb, lb)]["m"].append(m); buckets[(op, tb, lb)]["e"].append(e)
        op_all[op]["m"].append(m); op_all[op]["e"].append(e)

    fit = lambda M, E: _fit_group(M, E, costs, levels, br, bref)
    payload = {
        "stoc_len_levels": levels,
        "budget_ratio": br,
        "budget_ref_stoc_len": bref,
        "timestep_buckets": TB,
        "layer_buckets": LB,
        "operator_defaults": {},
        "buckets": {},
    }
    for op in sorted(operators):
        if not op_all[op]["m"]:
            continue
        payload["operator_defaults"][op] = fit(
            np.concatenate(op_all[op]["m"]), np.concatenate(op_all[op]["e"]))
    for key in sorted(buckets.keys()):
        M = np.concatenate(buckets[key]["m"]); E = np.concatenate(buckets[key]["e"])
        if M.size < min_bucket_units:
            continue
        op, tb, lb = key
        payload["buckets"][f"{op}:t{tb}:l{lb}"] = fit(M, E)
    payload["selected_timesteps"] = sorted(ts_seen, reverse=True)
    payload["operators"] = sorted(operators)

    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[rebucket] TB={TB} LB={LB}: {len(payload['buckets'])} buckets, "
          f"{len(ts_seen)} probed timesteps -> {out_json}")


if __name__ == "__main__":
    main()
