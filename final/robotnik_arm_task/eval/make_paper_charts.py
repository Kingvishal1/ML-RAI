#!/usr/bin/env python3
"""Generate paper-ready charts from evaluation JSONs.

Usage:
    python eval/make_paper_charts.py \
        --single-expert eval/results/expert_single_*.json \
        --single-learned eval/results/single_eval_*.json \
        --pnp-expert eval/results/expert_eval_*_120935.json \
        --pnp-learned eval/results/pnp_eval_*.json \
        --out eval/results/paper
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def load(path):
    with open(path) as f:
        return json.load(f)


def make_single_target_chart(expert, learned, out_path):
    """Fig 1: Single-target reaching — 2 panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#4c72b0", "#dd8452"]
    w = 0.35

    eb = np.array(expert["best_errs"])
    lb = np.array(learned["best_errs"])

    # 1. EE error distribution (zoom to 0–8mm)
    ax = axes[0]
    bins = np.linspace(0, 0.008, 40)
    ax.hist(eb, bins=bins, alpha=0.65, label="Expert", color=colors[0])
    ax.hist(lb, bins=bins, alpha=0.65, label="Learned", color=colors[1])
    ax.axvline(0.005, color="gray", linestyle="--", linewidth=0.8, label="5mm tolerance")
    ax.set_xlabel("Distance to target (m)")
    ax.set_ylabel("Count")
    ax.set_title("(a) End-Effector Precision")
    ax.legend(fontsize=8)

    # 2. Timing comparison
    ax = axes[1]
    e_times = expert["times"]
    l_times = learned["times"]
    t_cats = ["Mean", "Median", "P90"]
    e_t = [np.mean(e_times), np.median(e_times), np.percentile(e_times, 90)]
    l_t = [np.mean(l_times), np.median(l_times), np.percentile(l_times, 90)]
    x = np.arange(len(t_cats))
    bars_e = ax.bar(x - w / 2, e_t, w, label="Expert", color=colors[0])
    bars_l = ax.bar(x + w / 2, l_t, w, label="Learned", color=colors[1])
    for bars in [bars_e, bars_l]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(t_cats)
    ax.set_ylabel("Time (s)")
    ax.set_title("(b) Episode Timing")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = f"{out_path}_single.pdf"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def make_sweep_chart(out_path):
    """Fig: Combined phase-switching threshold sweeps — 2 panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import glob

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 1. Distance multiplier sweep
    ax = axes[0]
    scale_files = sorted(glob.glob("sweep_results/scale_*.json"))
    scales, final_errs = [], []
    for f in scale_files:
        d = load(f)
        s = d["threshold_scale"]
        if 0.75 <= s <= 0.95:
            scales.append(s)
            final_errs.append(d["final_ee_mean"] * 1000)
    ax.bar([str(s) for s in scales], final_errs, color="#4c72b0", width=0.55)
    for i, (s, v) in enumerate(zip(scales, final_errs)):
        ax.text(i, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    best_idx = final_errs.index(min(final_errs))
    ax.get_children()[best_idx].set_color("#dd8452")
    ax.set_xlabel("Distance multiplier")
    ax.set_ylabel("Mean final EE error (mm)")
    ax.set_title("(a) Distance Multiplier Sweep")

    # 2. Heading threshold sweep
    ax = axes[1]
    head_files = sorted(glob.glob("sweep_results/heading_*.json"))
    headings_deg, head_errs = [], []
    for f in head_files:
        d = load(f)
        headings_deg.append(round(d["heading_threshold_deg"]))
        head_errs.append(d["final_ee_mean"] * 1000)
    ax.bar([str(h) for h in headings_deg], head_errs, color="#4c72b0", width=0.55)
    for i, (h, v) in enumerate(zip(headings_deg, head_errs)):
        ax.text(i, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    best_idx = head_errs.index(min(head_errs))
    ax.get_children()[best_idx].set_color("#dd8452")
    ax.set_xlabel("Heading threshold (deg)")
    ax.set_ylabel("Mean final EE error (mm)")
    ax.set_title("(b) Heading Threshold Sweep")

    plt.tight_layout()
    path = f"{out_path}_sweeps.pdf"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def _collect_per_ball(data):
    balls = []
    for ep in data["per_episode"]:
        for i, b in enumerate(ep["per_ball"]):
            b = dict(b)
            b["_ep"] = ep["episode"]
            b["_ball_idx"] = i
            balls.append(b)
    return balls


def _is_gripped(b):
    v = b.get("gripped", 0)
    return bool(v) if isinstance(v, bool) else v >= 0.5


def _is_delivered(b):
    v = b.get("delivered", 0)
    return bool(v) if isinstance(v, bool) else v >= 0.5


def make_pnp_chart(expert, learned, out_path):
    """Fig 2: Pick-and-place — 2 panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#4c72b0", "#dd8452"]
    w = 0.35

    e_balls = _collect_per_ball(expert)
    l_balls = _collect_per_ball(learned)

    # 1. EE accuracy (pick & drop mean errors in mm)
    ax = axes[0]
    ee_cats = ["Pick\nprecision", "Drop\nprecision"]

    e_pick_final = [b["pick_final_err"] for b in e_balls if "pick_final_err" in b]
    e_gripped = [b for b in e_balls if _is_gripped(b)]
    e_drop_final = [b["drop_final_err"] for b in e_gripped if "drop_final_err" in b]

    l_pick_final = [b["pick_final_err"] for b in l_balls if "pick_final_err" in b]
    l_gripped = [b for b in l_balls if _is_gripped(b)]
    l_drop_final = [b["drop_final_err"] for b in l_gripped if "drop_final_err" in b]

    e_ee = [np.mean(e_pick_final) * 1000, np.mean(e_drop_final) * 1000]
    l_ee = [np.mean(l_pick_final) * 1000, np.mean(l_drop_final) * 1000]

    x = np.arange(len(ee_cats))
    bars_e = ax.bar(x - w / 2, e_ee, w, label="Expert", color=colors[0])
    bars_l = ax.bar(x + w / 2, l_ee, w, label="Learned", color=colors[1])
    for bars in [bars_e, bars_l]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ee_cats, fontsize=9)
    ax.set_ylabel("Mean distance to target (mm)")
    ax.set_title("(a) End-Effector Precision")
    ax.legend(fontsize=8)

    # 2. Timing
    ax = axes[1]
    e_times = [ep["total_time"] for ep in expert["per_episode"]]
    l_times = [ep["total_time"] for ep in learned["per_episode"]]
    t_cats = ["Mean", "Median", "P90"]
    e_t = [np.mean(e_times), np.median(e_times), np.percentile(e_times, 90)]
    l_t = [np.mean(l_times), np.median(l_times), np.percentile(l_times, 90)]
    x = np.arange(len(t_cats))
    bars_e = ax.bar(x - w / 2, e_t, w, label="Expert", color=colors[0])
    bars_l = ax.bar(x + w / 2, l_t, w, label="Learned", color=colors[1])
    for bars in [bars_e, bars_l]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(t_cats)
    ax.set_ylabel("Time (s)")
    ax.set_title("(b) Episode Timing")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = f"{out_path}_pnp.pdf"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready charts")
    parser.add_argument("--single-expert", help="Expert single-target JSON")
    parser.add_argument("--single-learned", help="Learned single-target JSON")
    parser.add_argument("--pnp-expert", help="Expert PnP JSON")
    parser.add_argument("--pnp-learned", help="Learned PnP JSON")
    parser.add_argument("--sweeps", action="store_true", help="Generate sweep chart")
    parser.add_argument("--out", default="eval/results/paper", help="Output prefix")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.single_expert and args.single_learned:
        print("Generating single-target chart...")
        make_single_target_chart(load(args.single_expert), load(args.single_learned), args.out)

    if args.pnp_expert and args.pnp_learned:
        print("Generating PnP chart...")
        make_pnp_chart(load(args.pnp_expert), load(args.pnp_learned), args.out)

    if args.sweeps:
        print("Generating sweep chart...")
        make_sweep_chart(args.out)


if __name__ == "__main__":
    main()
