#!/usr/bin/env python3
"""Compare expert vs learned policy evaluation runs side-by-side.

Usage:
    python eval/compare_runs.py <expert.json> <learned.json>
    python eval/compare_runs.py <expert.json> <learned.json> --out results/comparison
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


def _safe_mean(vals):
    return float(np.mean(vals)) if len(vals) else float("nan")


def _safe_median(vals):
    return float(np.median(vals)) if len(vals) else float("nan")


def _p90(vals):
    return float(np.percentile(vals, 90)) if len(vals) else float("nan")


def _ci95_binomial(k, n):
    """Wilson score 95% CI for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p_hat = k / n
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _mann_whitney(a, b):
    """Return (U, p) or None if scipy unavailable or data empty."""
    if len(a) == 0 or len(b) == 0:
        return None
    try:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        return (float(u), float(p))
    except ImportError:
        return None


def _bucket_threshold_counts(vals, thresholds):
    """Count values below each threshold. Returns list of (threshold, count, total)."""
    n = len(vals)
    arr = np.asarray(vals)
    return [(t, int(np.sum(arr < t)), n) for t in thresholds]


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _collect_per_ball(data):
    """Flatten per_episode.per_ball into a single list with episode index attached."""
    balls = []
    for ep in data["per_episode"]:
        for i, b in enumerate(ep["per_ball"]):
            b = dict(b)
            b["_ep"] = ep["episode"]
            b["_ball_idx"] = i
            balls.append(b)
    return balls


def _is_gripped(b, tolerance=None):
    if tolerance is not None:
        return b.get("grip_dist", float("inf")) < tolerance
    v = b.get("gripped", 0)
    return bool(v) if isinstance(v, bool) else v >= 0.5


def _is_delivered(b, tolerance=None):
    if tolerance is not None:
        # Must be gripped first AND drop_dist within new tolerance
        return _is_gripped(b, tolerance) and b.get("drop_dist", float("inf")) < tolerance
    v = b.get("delivered", 0)
    return bool(v) if isinstance(v, bool) else v >= 0.5


def _xy_dist(pos):
    return math.sqrt(pos[0] ** 2 + pos[1] ** 2)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(data, label, tolerance=None):
    """Return a dict of all metrics for one run.

    If tolerance is given (in metres), re-label gripped/delivered using
    grip_dist < tolerance and drop_dist < tolerance instead of the
    original boolean flags.
    """
    balls = _collect_per_ball(data)
    episodes = data["per_episode"]
    n_balls = len(balls)
    n_eps = len(episodes)

    gripped = [b for b in balls if _is_gripped(b, tolerance)]
    delivered = [b for b in balls if _is_delivered(b, tolerance)]

    m = {}
    m["label"] = label
    m["n_episodes"] = n_eps
    m["n_balls"] = n_balls

    # 1. Overall success rates
    m["grip_rate"] = _pct(len(gripped), n_balls)
    m["delivery_rate"] = _pct(len(delivered), n_balls)
    # Recompute per-episode delivery count when re-thresholding
    if tolerance is not None:
        def _ep_delivered(ep):
            return sum(1 for b in ep["per_ball"]
                       if _is_delivered(b, tolerance))
        full_del = sum(1 for ep in episodes
                       if _ep_delivered(ep) == ep["n_balls"])
    else:
        full_del = sum(1 for ep in episodes
                       if ep["balls_delivered"] == ep["n_balls"])
    m["full_delivery_rate"] = _pct(full_del, n_eps)
    m["grip_to_delivery"] = _pct(len(delivered), len(gripped)) if gripped else 0.0
    m["delivery_ci95"] = _ci95_binomial(len(delivered), n_balls)

    # 2. Per-ball-order breakdown
    m["per_ball_order"] = {}
    n_per_ep = data.get("num_targets_per_episode", 3)
    for idx in range(n_per_ep):
        subset = [b for b in balls if b["_ball_idx"] == idx]
        g = sum(1 for b in subset if _is_gripped(b, tolerance))
        d = sum(1 for b in subset if _is_delivered(b, tolerance))
        m["per_ball_order"][idx] = {
            "grip_rate": _pct(g, len(subset)),
            "delivery_rate": _pct(d, len(subset)),
            "n": len(subset),
        }

    # 3. Per-zone delivery rate
    m["per_zone"] = {}
    zones = sorted(set(b.get("drop_zone_idx", -1) for b in balls))
    for z in zones:
        subset = [b for b in balls if b.get("drop_zone_idx", -1) == z]
        d = sum(1 for b in subset if _is_delivered(b, tolerance))
        m["per_zone"][z] = {
            "delivery_rate": _pct(d, len(subset)),
            "n": len(subset),
        }

    # 4. Success by target difficulty
    def _bucket_balls(balls, key_fn, buckets):
        out = {}
        for name, lo, hi in buckets:
            subset = [b for b in balls if lo <= key_fn(b) < hi]
            g = sum(1 for b in subset if _is_gripped(b, tolerance))
            d = sum(1 for b in subset if _is_delivered(b, tolerance))
            out[name] = {"grip_rate": _pct(g, len(subset)),
                         "delivery_rate": _pct(d, len(subset)),
                         "n": len(subset)}
        return out

    m["by_xy_dist"] = _bucket_balls(
        balls,
        lambda b: _xy_dist(b["ball_pos"]),
        [("near(<2.5m)", 0, 2.5), ("mid(2.5-4m)", 2.5, 4.0), ("far(>4m)", 4.0, 100)],
    )
    m["by_height"] = _bucket_balls(
        balls,
        lambda b: b["ball_pos"][2],
        [("low(<0.7)", 0, 0.7), ("high(>=0.7)", 0.7, 100)],
    )

    # 5. EE accuracy
    pick_best = [b["pick_best_err"] for b in balls if "pick_best_err" in b]
    pick_final = [b["pick_final_err"] for b in balls if "pick_final_err" in b]
    drop_best = [b["drop_best_err"] for b in gripped if "drop_best_err" in b]
    drop_final = [b["drop_final_err"] for b in gripped if "drop_final_err" in b]

    ee_thresholds = [0.005, 0.01, 0.05, 0.1]
    m["ee_accuracy"] = {}
    for name, vals in [("pick_best", pick_best), ("pick_final", pick_final),
                       ("drop_best", drop_best), ("drop_final", drop_final)]:
        m["ee_accuracy"][name] = {
            "mean": _safe_mean(vals), "median": _safe_median(vals), "p90": _p90(vals),
            "thresholds": {f"<{t}m": int(np.sum(np.array(vals) < t))
                           for t in ee_thresholds} if vals else {},
            "n": len(vals),
        }

    # 6. Grip & drop distance
    grip_d = [b["grip_dist"] for b in balls if "grip_dist" in b]
    drop_d = [b["drop_dist"] for b in gripped if "drop_dist" in b]
    for name, vals in [("grip_dist", grip_d), ("drop_dist", drop_d)]:
        m[name] = {
            "mean": _safe_mean(vals), "median": _safe_median(vals),
            "p90": _p90(vals), "max": float(np.max(vals)) if vals else float("nan"),
            "n": len(vals),
        }

    # 7. Timing
    times = [ep["total_time"] for ep in episodes]
    steps = [ep["total_steps"] for ep in episodes]
    m["time_per_ep"] = {"mean": _safe_mean(times), "median": _safe_median(times), "p90": _p90(times)}
    m["steps_per_ep"] = {"mean": _safe_mean(steps), "median": _safe_median(steps), "p90": _p90(steps)}
    n_per = data.get("num_targets_per_episode", 3)
    time_per_ball = [t / n_per for t in times]
    m["time_per_ball"] = {"mean": _safe_mean(time_per_ball), "median": _safe_median(time_per_ball)}

    # 8. Selector / path efficiency
    sel = data.get("selector_dists", [])
    exp = data.get("expert_dists", [])
    oh = data.get("dist_overheads_pct", [])
    m["selector_dist"] = {"mean": _safe_mean(sel), "median": _safe_median(sel)}
    m["expert_dist"] = {"mean": _safe_mean(exp), "median": _safe_median(exp)}
    m["dist_overhead_pct"] = {"mean": _safe_mean(oh), "median": _safe_median(oh)}
    order_match = sum(
        1 for ep in episodes if ep.get("visit_order") == ep.get("expert_order")
    )
    m["order_match_rate"] = _pct(order_match, n_eps)

    # 9. Failure analysis
    grip_fail = sum(1 for b in balls if not _is_gripped(b, tolerance))
    drop_fail = sum(1 for b in gripped if not _is_delivered(b, tolerance))
    m["failures"] = {
        "grip_fail": grip_fail,
        "grip_fail_pct": _pct(grip_fail, n_balls),
        "drop_fail": drop_fail,
        "drop_fail_pct": _pct(drop_fail, len(gripped)) if gripped else 0.0,
    }
    # Worst episodes (recompute delivery count when re-thresholding)
    def _ep_del_count(ep):
        if tolerance is not None:
            return sum(1 for b in ep["per_ball"] if _is_delivered(b, tolerance))
        return ep["balls_delivered"]
    worst = sorted(episodes, key=lambda e: _ep_del_count(e) / max(e["n_balls"], 1))
    m["worst_episodes"] = [
        {"episode": e["episode"],
         "delivered": _ep_del_count(e),
         "n_balls": e["n_balls"]}
        for e in worst[:5]
    ]

    # Raw arrays for statistical tests
    m["_raw"] = {
        "grip_dists": grip_d,
        "drop_dists": drop_d,
        "times": times,
        "steps": steps,
        "pick_final_errs": pick_final,
        "drop_final_errs": drop_final,
    }

    return m


# ---------------------------------------------------------------------------
# Comparison & formatting
# ---------------------------------------------------------------------------

def _delta_str(expert_val, learned_val, fmt=".2f", higher_is_better=True):
    """Format: expert | learned | delta (colored direction)."""
    d = learned_val - expert_val
    sign = "+" if d > 0 else ""
    arrow = ""
    if not math.isnan(d) and d != 0:
        better = (d > 0) == higher_is_better
        arrow = " ✓" if better else " ✗"
    return f"{expert_val:{fmt}}  |  {learned_val:{fmt}}  |  {sign}{d:{fmt}}{arrow}"


def _header(title):
    w = 70
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")


def _row(label, expert_val, learned_val, fmt=".2f", higher_is_better=True):
    print(f"  {label:30s}  {_delta_str(expert_val, learned_val, fmt, higher_is_better)}")


def print_comparison(me, ml, tolerance=None):
    """Print formatted comparison to stdout."""
    print("\n" + "=" * 70)
    print("  EVALUATION COMPARISON: Expert vs Learned Policy")
    print("=" * 70)
    tol_str = f"  Grip tolerance: {tolerance*1000:.0f}mm" if tolerance else "  Grip tolerance: original (100mm)"
    print(f"  Episodes: {me['n_episodes']}  |  Balls per episode: {me['n_balls'] // me['n_episodes']}")
    print(tol_str)
    print(f"  {'':30s}  {'Expert':>8s}  |  {'Learned':>8s}  |  {'Delta':>8s}")
    print(f"  {'-' * 64}")

    # 1. Overall success
    _header("1. Overall Success Rates (%)")
    _row("Grip rate", me["grip_rate"], ml["grip_rate"])
    _row("Delivery rate", me["delivery_rate"], ml["delivery_rate"])
    _row("Full episode delivery", me["full_delivery_rate"], ml["full_delivery_rate"])
    _row("Grip→delivery conversion", me["grip_to_delivery"], ml["grip_to_delivery"])

    # CI
    eci = me["delivery_ci95"]
    lci = ml["delivery_ci95"]
    print(f"  {'Delivery 95% CI':30s}  [{eci[0]:.3f}, {eci[1]:.3f}]  |  [{lci[0]:.3f}, {lci[1]:.3f}]")

    # 2. Per-ball order
    _header("2. Per-Ball-Order Breakdown (%)")
    for idx in sorted(me["per_ball_order"]):
        eb = me["per_ball_order"][idx]
        lb = ml["per_ball_order"][idx]
        print(f"  Ball #{idx + 1}:")
        _row(f"  Grip rate", eb["grip_rate"], lb["grip_rate"])
        _row(f"  Delivery rate", eb["delivery_rate"], lb["delivery_rate"])

    # 3. Per-zone
    _header("3. Per-Zone Delivery Rate (%)")
    all_zones = sorted(set(list(me["per_zone"].keys()) + list(ml["per_zone"].keys())))
    for z in all_zones:
        ez = me["per_zone"].get(z, {"delivery_rate": 0, "n": 0})
        lz = ml["per_zone"].get(z, {"delivery_rate": 0, "n": 0})
        _row(f"Zone {z} (n={ez['n']}/{lz['n']})",
             ez["delivery_rate"], lz["delivery_rate"])

    # 4. Difficulty buckets
    _header("4. Success by Target Difficulty (%)")
    print("  By XY distance from origin:")
    for key in ["near(<2.5m)", "mid(2.5-4m)", "far(>4m)"]:
        eb = me["by_xy_dist"].get(key, {"grip_rate": 0, "delivery_rate": 0, "n": 0})
        lb = ml["by_xy_dist"].get(key, {"grip_rate": 0, "delivery_rate": 0, "n": 0})
        _row(f"  {key} grip", eb["grip_rate"], lb["grip_rate"])
        _row(f"  {key} delivery", eb["delivery_rate"], lb["delivery_rate"])
    print("  By ball height:")
    for key in ["low(<0.7)", "high(>=0.7)"]:
        eb = me["by_height"].get(key, {"grip_rate": 0, "delivery_rate": 0, "n": 0})
        lb = ml["by_height"].get(key, {"grip_rate": 0, "delivery_rate": 0, "n": 0})
        _row(f"  {key} grip", eb["grip_rate"], lb["grip_rate"])
        _row(f"  {key} delivery", eb["delivery_rate"], lb["delivery_rate"])

    # 5. EE accuracy
    _header("5. End-Effector Accuracy (m)")
    for name in ["pick_best", "pick_final", "drop_best", "drop_final"]:
        ee = me["ee_accuracy"][name]
        el = ml["ee_accuracy"][name]
        print(f"  {name}:")
        _row("    mean", ee["mean"], el["mean"], ".6f", higher_is_better=False)
        _row("    median", ee["median"], el["median"], ".6f", higher_is_better=False)
        _row("    p90", ee["p90"], el["p90"], ".6f", higher_is_better=False)
        # threshold counts
        for t_key in ee.get("thresholds", {}):
            ec = ee["thresholds"].get(t_key, 0)
            lc = el["thresholds"].get(t_key, 0)
            en, ln = ee["n"], el["n"]
            print(f"    {'  ' + t_key:30s}  {ec}/{en} ({_pct(ec,en):.0f}%)  |  "
                  f"{lc}/{ln} ({_pct(lc,ln):.0f}%)")

    # 6. Grip & drop distance
    _header("6. Grip & Drop Distance (m)")
    for name in ["grip_dist", "drop_dist"]:
        ed = me[name]
        ld = ml[name]
        print(f"  {name}:")
        _row("    mean", ed["mean"], ld["mean"], ".6f", higher_is_better=False)
        _row("    median", ed["median"], ld["median"], ".6f", higher_is_better=False)
        _row("    p90", ed["p90"], ld["p90"], ".6f", higher_is_better=False)
        _row("    max", ed["max"], ld["max"], ".6f", higher_is_better=False)

    # 7. Timing
    _header("7. Timing")
    print("  Time per episode (s):")
    _row("    mean", me["time_per_ep"]["mean"], ml["time_per_ep"]["mean"], ".1f", False)
    _row("    median", me["time_per_ep"]["median"], ml["time_per_ep"]["median"], ".1f", False)
    _row("    p90", me["time_per_ep"]["p90"], ml["time_per_ep"]["p90"], ".1f", False)
    print("  Steps per episode:")
    _row("    mean", me["steps_per_ep"]["mean"], ml["steps_per_ep"]["mean"], ".0f", False)
    print("  Time per ball (s):")
    _row("    mean", me["time_per_ball"]["mean"], ml["time_per_ball"]["mean"], ".1f", False)

    # 8. Path efficiency
    _header("8. Selector / Path Efficiency")
    _row("Selector dist (mean)", me["selector_dist"]["mean"], ml["selector_dist"]["mean"], ".2f", False)
    _row("Expert dist (mean)", me["expert_dist"]["mean"], ml["expert_dist"]["mean"], ".2f", False)
    _row("Overhead % (mean)", me["dist_overhead_pct"]["mean"], ml["dist_overhead_pct"]["mean"], ".2f", False)
    _row("Order match rate %", me["order_match_rate"], ml["order_match_rate"])

    # 9. Failure analysis
    _header("9. Failure Analysis")
    ef = me["failures"]
    lf = ml["failures"]
    _row("Grip failures", ef["grip_fail"], lf["grip_fail"], ".0f", False)
    _row("Grip fail %", ef["grip_fail_pct"], lf["grip_fail_pct"], ".1f", False)
    _row("Drop failures (gripped!del)", ef["drop_fail"], lf["drop_fail"], ".0f", False)
    _row("Drop fail %", ef["drop_fail_pct"], lf["drop_fail_pct"], ".1f", False)

    print("\n  Worst episodes (expert):")
    for w in me["worst_episodes"]:
        print(f"    ep {w['episode']:3d}: {w['delivered']}/{w['n_balls']} delivered")
    print("  Worst episodes (learned):")
    for w in ml["worst_episodes"]:
        print(f"    ep {w['episode']:3d}: {w['delivered']}/{w['n_balls']} delivered")

    # 10. Statistical comparison
    _header("10. Statistical Tests (Mann-Whitney U)")
    for key, label in [("grip_dists", "Grip distance"),
                       ("drop_dists", "Drop distance"),
                       ("times", "Episode time"),
                       ("pick_final_errs", "Pick final err"),
                       ("drop_final_errs", "Drop final err")]:
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
    """Create a 2x3 comparison chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip chart — matplotlib not installed]")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Expert vs Learned Policy Comparison", fontsize=14, fontweight="bold")
    colors = ["#4c72b0", "#dd8452"]
    labels_pair = ["Expert", "Learned"]

    # 1. Success rates (bars)
    ax = axes[0, 0]
    cats = ["Grip", "Delivery", "Full Ep.", "Grip→Del"]
    e_vals = [me["grip_rate"], me["delivery_rate"], me["full_delivery_rate"], me["grip_to_delivery"]]
    l_vals = [ml["grip_rate"], ml["delivery_rate"], ml["full_delivery_rate"], ml["grip_to_delivery"]]
    x = np.arange(len(cats))
    w = 0.35
    ax.bar(x - w / 2, e_vals, w, label=labels_pair[0], color=colors[0])
    ax.bar(x + w / 2, l_vals, w, label=labels_pair[1], color=colors[1])
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("%")
    ax.set_title("Success Rates")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)

    # 2. Grip distance histogram
    ax = axes[0, 1]
    eg = me["_raw"]["grip_dists"]
    lg = ml["_raw"]["grip_dists"]
    bins = np.linspace(0, max(max(eg) if eg else 0.1, max(lg) if lg else 0.1) * 1.05, 30)
    ax.hist(eg, bins=bins, alpha=0.6, label="Expert", color=colors[0])
    ax.hist(lg, bins=bins, alpha=0.6, label="Learned", color=colors[1])
    ax.set_xlabel("Grip distance (m)")
    ax.set_ylabel("Count")
    ax.set_title("Grip Distance Distribution")
    ax.legend(fontsize=8)

    # 3. Drop distance histogram
    ax = axes[0, 2]
    ed = me["_raw"]["drop_dists"]
    ld = ml["_raw"]["drop_dists"]
    bins = np.linspace(0, max(max(ed) if ed else 0.1, max(ld) if ld else 0.1) * 1.05, 30)
    ax.hist(ed, bins=bins, alpha=0.6, label="Expert", color=colors[0])
    ax.hist(ld, bins=bins, alpha=0.6, label="Learned", color=colors[1])
    ax.set_xlabel("Drop distance (m)")
    ax.set_ylabel("Count")
    ax.set_title("Drop Distance Distribution")
    ax.legend(fontsize=8)

    # 4. EE accuracy comparison (grouped bars)
    ax = axes[1, 0]
    ee_cats = ["pick_best", "pick_final", "drop_best", "drop_final"]
    ee_labels = ["Pick Best", "Pick Final", "Drop Best", "Drop Final"]
    e_ee = [me["ee_accuracy"][k]["mean"] * 1000 for k in ee_cats]  # mm
    l_ee = [ml["ee_accuracy"][k]["mean"] * 1000 for k in ee_cats]
    x = np.arange(len(ee_cats))
    ax.bar(x - w / 2, e_ee, w, label="Expert", color=colors[0])
    ax.bar(x + w / 2, l_ee, w, label="Learned", color=colors[1])
    ax.set_xticks(x)
    ax.set_xticklabels(ee_labels, fontsize=9)
    ax.set_ylabel("Mean error (mm)")
    ax.set_title("EE Accuracy")
    ax.legend(fontsize=8)

    # 5. Per-ball-order success
    ax = axes[1, 1]
    ball_idxs = sorted(me["per_ball_order"])
    ball_labels = [f"Ball #{i + 1}" for i in ball_idxs]
    e_del = [me["per_ball_order"][i]["delivery_rate"] for i in ball_idxs]
    l_del = [ml["per_ball_order"][i]["delivery_rate"] for i in ball_idxs]
    x = np.arange(len(ball_idxs))
    ax.bar(x - w / 2, e_del, w, label="Expert", color=colors[0])
    ax.bar(x + w / 2, l_del, w, label="Learned", color=colors[1])
    ax.set_xticks(x)
    ax.set_xticklabels(ball_labels)
    ax.set_ylabel("Delivery rate (%)")
    ax.set_title("Per-Ball-Order Delivery")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)

    # 6. Timing comparison
    ax = axes[1, 2]
    t_cats = ["Mean", "Median", "P90"]
    e_t = [me["time_per_ep"]["mean"], me["time_per_ep"]["median"], me["time_per_ep"]["p90"]]
    l_t = [ml["time_per_ep"]["mean"], ml["time_per_ep"]["median"], ml["time_per_ep"]["p90"]]
    x = np.arange(len(t_cats))
    ax.bar(x - w / 2, e_t, w, label="Expert", color=colors[0])
    ax.bar(x + w / 2, l_t, w, label="Learned", color=colors[1])
    ax.set_xticks(x)
    ax.set_xticklabels(t_cats)
    ax.set_ylabel("Time (s)")
    ax.set_title("Episode Timing")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved to {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def _strip_raw(m):
    """Return metrics dict without internal _raw arrays."""
    return {k: v for k, v in m.items() if k != "_raw"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare expert vs learned eval runs")
    parser.add_argument("expert_json", help="Path to expert evaluation JSON")
    parser.add_argument("learned_json", help="Path to learned policy evaluation JSON")
    parser.add_argument("--out", default=None,
                        help="Output prefix (produces <prefix>_comparison.json and <prefix>_comparison.png)")
    parser.add_argument("--grip_tolerance", type=float, default=None,
                        help="Override grip/drop tolerance in metres (e.g. 0.05 for 50mm). "
                             "Re-labels success using raw distances from the saved data.")
    args = parser.parse_args()

    tol = args.grip_tolerance

    with open(args.expert_json) as f:
        expert_data = json.load(f)
    with open(args.learned_json) as f:
        learned_data = json.load(f)

    me = compute_metrics(expert_data, "expert", tolerance=tol)
    ml = compute_metrics(learned_data, "learned", tolerance=tol)

    print_comparison(me, ml, tolerance=tol)

    # Determine output prefix
    if args.out:
        out_prefix = args.out
    else:
        out_prefix = os.path.join(
            os.path.dirname(args.learned_json) or ".",
            "comparison",
        )

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    # Save JSON
    json_path = f"{out_prefix}_comparison.json"
    combined = {
        "grip_tolerance": tol if tol else 0.10,
        "expert": _strip_raw(me),
        "learned": _strip_raw(ml),
        "summary": {
            "grip_rate_delta": ml["grip_rate"] - me["grip_rate"],
            "delivery_rate_delta": ml["delivery_rate"] - me["delivery_rate"],
            "full_delivery_delta": ml["full_delivery_rate"] - me["full_delivery_rate"],
            "time_per_ep_delta": ml["time_per_ep"]["mean"] - me["time_per_ep"]["mean"],
        },
    }
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"  Metrics saved to {json_path}")

    # Save chart
    chart_path = f"{out_prefix}_comparison.png"
    make_chart(me, ml, chart_path)


if __name__ == "__main__":
    main()
