#!/usr/bin/env python3
"""Compare expert vs learned single-target reaching evaluations.

Usage:
    python eval/compare_single.py <expert.json> <learned.json>
    python eval/compare_single.py <expert.json> <learned.json> --out eval/results/single_cmp
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import print_stats, print_buckets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(num, den):
    return 100.0 * num / den if den else 0.0


def _safe_mean(v):
    return float(np.mean(v)) if len(v) else float("nan")


def _safe_median(v):
    return float(np.median(v)) if len(v) else float("nan")


def _p90(v):
    return float(np.percentile(v, 90)) if len(v) else float("nan")


def _p10(v):
    return float(np.percentile(v, 10)) if len(v) else float("nan")


def _ci95_binomial(k, n):
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p_hat = k / n
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _mann_whitney(a, b):
    if len(a) == 0 or len(b) == 0:
        return None
    try:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        return (float(u), float(p))
    except ImportError:
        return None


def _xy_dist(pos):
    return math.sqrt(pos[0] ** 2 + pos[1] ** 2)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(data, label, tolerance=None):
    m = {}
    m["label"] = label
    n = data["num_targets"]
    m["n_targets"] = n

    best = np.array(data["best_errs"])
    final = np.array(data["final_errs"])
    times = np.array(data["times"])

    tol = tolerance if tolerance is not None else data.get("reach_tolerance", 0.05)
    reached = int(np.sum(best < tol))
    m["reach_tolerance"] = tol

    # 1. Reach rate
    m["reach_rate"] = _pct(reached, n)
    m["reach_ci95"] = _ci95_binomial(reached, n)

    # 2. EE error
    ee_thresholds = [0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    for name, arr in [("best_err", best), ("final_err", final)]:
        m[name] = {
            "mean": _safe_mean(arr),
            "median": _safe_median(arr),
            "p90": _p90(arr),
            "p10": _p10(arr),
            "max": float(np.max(arr)) if len(arr) else float("nan"),
            "min": float(np.min(arr)) if len(arr) else float("nan"),
            "thresholds": {f"<{t}m": int(np.sum(arr < t)) for t in ee_thresholds},
            "n": len(arr),
        }

    # 3. Timing
    m["time"] = {
        "mean": _safe_mean(times),
        "median": _safe_median(times),
        "p90": _p90(times),
    }

    # 4. Per-phase (learned only)
    for key in ["base_best_errs", "base_final_errs", "arm_best_errs", "arm_final_errs"]:
        arr = np.array(data.get(key, []))
        if len(arr):
            m[key] = {"mean": _safe_mean(arr), "median": _safe_median(arr), "p90": _p90(arr), "n": len(arr)}

    # Base positioning accuracy (learned only)
    xy_errs = np.array(data.get("base_xy_errs", []))
    bearing_errs = np.array(data.get("base_bearing_errs", []))
    if len(xy_errs):
        m["base_xy_err"] = {"mean": _safe_mean(xy_errs), "median": _safe_median(xy_errs), "p90": _p90(xy_errs)}
    if len(bearing_errs):
        bearing_degs = np.degrees(bearing_errs)
        m["base_bearing_err_deg"] = {"mean": _safe_mean(bearing_degs), "median": _safe_median(bearing_degs), "p90": _p90(bearing_degs)}

    switch_times = np.array(data.get("switch_times", []))
    if len(switch_times):
        m["switch_time"] = {"mean": _safe_mean(switch_times), "median": _safe_median(switch_times)}

    # 5. By target difficulty (if per_target has target positions)
    per_target = data.get("per_target", [])
    if per_target and "target" in per_target[0]:
        for bucket_name, lo, hi in [("near(<2.5m)", 0, 2.5), ("mid(2.5-4m)", 2.5, 4.0), ("far(>4m)", 4.0, 100)]:
            subset = [t for t in per_target if lo <= _xy_dist(t["target"]) < hi]
            if subset:
                sub_best = [t["best_err"] for t in subset]
                sub_reached = sum(1 for e in sub_best if e < tol)
                m[f"by_dist_{bucket_name}"] = {
                    "n": len(subset),
                    "reach_rate": _pct(sub_reached, len(subset)),
                    "mean_best": _safe_mean(sub_best),
                }

        for bucket_name, lo, hi in [("low(<0.7)", 0, 0.7), ("high(>=0.7)", 0.7, 100)]:
            subset = [t for t in per_target if lo <= t["target"][2] < hi]
            if subset:
                sub_best = [t["best_err"] for t in subset]
                sub_reached = sum(1 for e in sub_best if e < tol)
                m[f"by_height_{bucket_name}"] = {
                    "n": len(subset),
                    "reach_rate": _pct(sub_reached, len(subset)),
                    "mean_best": _safe_mean(sub_best),
                }

    # Raw arrays for stats tests
    m["_raw"] = {"best_errs": best.tolist(), "final_errs": final.tolist(), "times": times.tolist()}
    return m


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _delta_str(ev, lv, fmt=".4f", higher_is_better=True):
    d = lv - ev
    sign = "+" if d > 0 else ""
    arrow = ""
    if not math.isnan(d) and d != 0:
        better = (d > 0) == higher_is_better
        arrow = " ✓" if better else " ✗"
    return f"{ev:{fmt}}  |  {lv:{fmt}}  |  {sign}{d:{fmt}}{arrow}"


def _header(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _row(label, ev, lv, fmt=".4f", higher_is_better=True):
    print(f"  {label:30s}  {_delta_str(ev, lv, fmt, higher_is_better)}")


def print_comparison(me, ml):
    print("\n" + "=" * 70)
    print("  SINGLE-TARGET REACHING: Expert vs Learned")
    print("=" * 70)
    print(f"  Targets: {me['n_targets']}  |  Reach tolerance: {me['reach_tolerance'] * 1000:.0f}mm")
    print(f"  {'':30s}  {'Expert':>8s}  |  {'Learned':>8s}  |  {'Delta':>8s}")
    print(f"  {'-' * 64}")

    # 1. Reach rate
    _header("1. Reach Rate (%)")
    _row("Reach rate", me["reach_rate"], ml["reach_rate"], ".2f")
    eci = me["reach_ci95"]
    lci = ml["reach_ci95"]
    print(f"  {'95% CI':30s}  [{eci[0]:.3f}, {eci[1]:.3f}]  |  [{lci[0]:.3f}, {lci[1]:.3f}]")

    # 2. EE error
    _header("2. End-Effector Error (m)")
    for name, label in [("best_err", "Best EE error"), ("final_err", "Final EE error")]:
        print(f"  {label}:")
        _row("    mean", me[name]["mean"], ml[name]["mean"], ".6f", False)
        _row("    median", me[name]["median"], ml[name]["median"], ".6f", False)
        _row("    p90", me[name]["p90"], ml[name]["p90"], ".6f", False)
        _row("    max", me[name]["max"], ml[name]["max"], ".6f", False)
        # Threshold counts
        for t_key in me[name].get("thresholds", {}):
            ec = me[name]["thresholds"][t_key]
            lc = ml[name]["thresholds"].get(t_key, 0)
            en, ln = me[name]["n"], ml[name]["n"]
            print(f"    {'  ' + t_key:30s}  {ec}/{en} ({_pct(ec,en):.0f}%)  |  "
                  f"{lc}/{ln} ({_pct(lc,ln):.0f}%)")

    # 3. Timing
    _header("3. Timing (s)")
    _row("Mean time", me["time"]["mean"], ml["time"]["mean"], ".1f", False)
    _row("Median time", me["time"]["median"], ml["time"]["median"], ".1f", False)
    _row("P90 time", me["time"]["p90"], ml["time"]["p90"], ".1f", False)

    # 4. Per-phase (learned only)
    if any(k in ml for k in ["base_best_errs", "arm_best_errs"]):
        _header("4. Per-Phase Breakdown (Learned only)")
        for key, label in [("base_best_errs", "Base best EE"),
                           ("base_final_errs", "Base final EE"),
                           ("arm_best_errs", "Arm best EE"),
                           ("arm_final_errs", "Arm final EE")]:
            if key in ml:
                d = ml[key]
                print(f"  {label:30s}  mean={d['mean']:.4f}  median={d['median']:.4f}  "
                      f"p90={d['p90']:.4f}  (n={d['n']})")
        if "base_xy_err" in ml:
            d = ml["base_xy_err"]
            print(f"  {'Base XY err vs standoff':30s}  mean={d['mean']:.4f}  median={d['median']:.4f}  p90={d['p90']:.4f}")
        if "base_bearing_err_deg" in ml:
            d = ml["base_bearing_err_deg"]
            print(f"  {'Base bearing err (deg)':30s}  mean={d['mean']:.1f}  median={d['median']:.1f}  p90={d['p90']:.1f}")
        if "switch_time" in ml:
            d = ml["switch_time"]
            print(f"  {'Phase switch time (s)':30s}  mean={d['mean']:.1f}  median={d['median']:.1f}")

    # 5. By difficulty (only if both sides have target data)
    dist_keys = [k for k in me if k.startswith("by_dist_") and k in ml]
    height_keys = [k for k in me if k.startswith("by_height_") and k in ml]
    if dist_keys or height_keys:
        _header("5. Reach Rate by Difficulty (%)")
        if dist_keys:
            print("  By XY distance from origin:")
            for k in sorted(dist_keys):
                bucket = k.replace("by_dist_", "")
                ed = me.get(k, {"reach_rate": 0, "n": 0, "mean_best": 0})
                ld = ml.get(k, {"reach_rate": 0, "n": 0, "mean_best": 0})
                _row(f"  {bucket} reach", ed["reach_rate"], ld["reach_rate"], ".1f")
                _row(f"  {bucket} mean best", ed["mean_best"], ld["mean_best"], ".4f", False)
        if height_keys:
            print("  By target height:")
            for k in sorted(height_keys):
                bucket = k.replace("by_height_", "")
                ed = me.get(k, {"reach_rate": 0, "n": 0, "mean_best": 0})
                ld = ml.get(k, {"reach_rate": 0, "n": 0, "mean_best": 0})
                _row(f"  {bucket} reach", ed["reach_rate"], ld["reach_rate"], ".1f")
                _row(f"  {bucket} mean best", ed["mean_best"], ld["mean_best"], ".4f", False)

    # 6. Statistical tests
    _header("6. Statistical Tests (Mann-Whitney U)")
    for key, label in [("best_errs", "Best EE error"),
                       ("final_errs", "Final EE error"),
                       ("times", "Time")]:
        result = _mann_whitney(
            np.array(me["_raw"][key]),
            np.array(ml["_raw"][key]),
        )
        if result:
            u, p = result
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            print(f"  {label:25s}  U={u:10.0f}  p={p:.4e}  {sig}")
        else:
            print(f"  {label:25s}  (scipy not available)")

    print()


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def make_chart(me, ml, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip chart — matplotlib not installed]")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Single-Target Reaching: Expert vs Learned", fontsize=14, fontweight="bold")
    colors = ["#4c72b0", "#dd8452"]
    w = 0.35

    # 1. Reach rate and threshold counts
    ax = axes[0]
    thresholds = ["<0.01m", "<0.05m", "<0.1m", "<0.2m", "<0.5m"]
    e_vals = [_pct(me["best_err"]["thresholds"].get(t, 0), me["best_err"]["n"]) for t in thresholds]
    l_vals = [_pct(ml["best_err"]["thresholds"].get(t, 0), ml["best_err"]["n"]) for t in thresholds]
    x = np.arange(len(thresholds))
    ax.bar(x - w / 2, e_vals, w, label="Expert", color=colors[0])
    ax.bar(x + w / 2, l_vals, w, label="Learned", color=colors[1])
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds, fontsize=9)
    ax.set_ylabel("% of targets")
    ax.set_title("Best EE Error — Cumulative %")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)

    # 2. EE error distribution (overlaid histograms)
    ax = axes[1]
    eb = np.array(me["_raw"]["best_errs"])
    lb = np.array(ml["_raw"]["best_errs"])
    upper = min(max(np.percentile(eb, 95), np.percentile(lb, 95)) * 1.2, 1.0)
    bins = np.linspace(0, upper, 30)
    ax.hist(eb, bins=bins, alpha=0.6, label="Expert", color=colors[0])
    ax.hist(lb, bins=bins, alpha=0.6, label="Learned", color=colors[1])
    ax.set_xlabel("Best EE error (m)")
    ax.set_ylabel("Count")
    ax.set_title("Best EE Error Distribution")
    ax.legend(fontsize=8)

    # 3. Time distribution
    ax = axes[2]
    et = np.array(me["_raw"]["times"])
    lt = np.array(ml["_raw"]["times"])
    upper_t = max(np.percentile(et, 95), np.percentile(lt, 95)) * 1.2
    bins_t = np.linspace(0, upper_t, 30)
    ax.hist(et, bins=bins_t, alpha=0.6, label="Expert", color=colors[0])
    ax.hist(lt, bins=bins_t, alpha=0.6, label="Learned", color=colors[1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count")
    ax.set_title("Episode Time Distribution")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved to {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare expert vs learned single-target reaching")
    parser.add_argument("expert_json", help="Path to expert single-target JSON")
    parser.add_argument("learned_json", help="Path to learned single-target JSON")
    parser.add_argument("--out", default=None,
                        help="Output prefix (produces <prefix>_comparison.json and .png)")
    parser.add_argument("--reach_tolerance", type=float, default=None,
                        help="Override reach tolerance in metres (e.g. 0.051 for 51mm)")
    args = parser.parse_args()

    tol = args.reach_tolerance

    with open(args.expert_json) as f:
        expert_data = json.load(f)
    with open(args.learned_json) as f:
        learned_data = json.load(f)

    me = compute_metrics(expert_data, "expert", tolerance=tol)
    ml = compute_metrics(learned_data, "learned", tolerance=tol)

    print_comparison(me, ml)

    if args.out:
        out_prefix = args.out
    else:
        out_prefix = os.path.join(
            os.path.dirname(args.learned_json) or ".", "single_comparison")

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    # JSON
    json_path = f"{out_prefix}_comparison.json"
    def _strip(m):
        return {k: v for k, v in m.items() if k != "_raw"}

    combined = {
        "expert": _strip(me),
        "learned": _strip(ml),
        "summary": {
            "reach_rate_delta": ml["reach_rate"] - me["reach_rate"],
            "best_err_mean_delta": ml["best_err"]["mean"] - me["best_err"]["mean"],
            "final_err_mean_delta": ml["final_err"]["mean"] - me["final_err"]["mean"],
            "time_mean_delta": ml["time"]["mean"] - me["time"]["mean"],
        },
    }
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"  Metrics saved to {json_path}")

    # Chart
    chart_path = f"{out_prefix}_comparison.png"
    make_chart(me, ml, chart_path)


if __name__ == "__main__":
    main()
