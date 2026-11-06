#!/usr/bin/env python3
"""Parse pick-and-place evaluation logs and generate metrics + charts.

Usage:
  python parse_eval_log.py eval.txt
  python parse_eval_log.py eval.txt --out results/my_run
"""

import argparse
import json
import os
import re
import sys

import numpy as np

from utils import print_stats, print_buckets


def parse_log(path):
    with open(path) as f:
        text = f.read()

    episodes = []
    current_ep = None
    current_balls = []

    for line in text.splitlines():
        # Episode start
        m = re.match(r'\[Episode\s+(\d+)/(\d+)\]', line)
        if m:
            if current_ep is not None:
                current_ep["per_ball"] = current_balls
                episodes.append(current_ep)
            current_ep = {"episode": int(m.group(1)), "total_episodes": int(m.group(2)),
                          "per_ball": []}
            current_balls = []
            continue

        # Selector chose ball
        m = re.search(r'Selector chose ball (\d+).*?\((\[[\d\s\.\-,]+\])\).*?zone (\d+)', line)
        if m:
            current_balls.append({
                "slot": int(m.group(1)),
                "zone": int(m.group(3)),
                "gripped": False,
                "delivered": False,
                "grip_dist": None,
                "drop_dist": None,
            })
            continue

        # Auto-gripped
        m = re.search(r'Auto-gripped \(dist=([\d.]+)m\)', line)
        if m and current_balls:
            current_balls[-1]["gripped"] = True
            current_balls[-1]["grip_dist"] = float(m.group(1))
            continue

        # Grip failed
        m = re.search(r'Grip failed \(dist=([\d.]+)m\)', line)
        if m and current_balls:
            current_balls[-1]["gripped"] = False
            current_balls[-1]["grip_dist"] = float(m.group(1))
            continue

        # Released at drop zone
        m = re.search(r'Released at drop zone \(dist=([\d.]+)m, delivered=(True|False)\)', line)
        if m and current_balls:
            current_balls[-1]["drop_dist"] = float(m.group(1))
            current_balls[-1]["delivered"] = m.group(2) == "True"
            continue

        # PnP complete line
        m = re.search(r'(\d+)/(\d+) gripped.*?(\d+)/(\d+) delivered', line)
        if m and current_ep is not None:
            current_ep["n_gripped"] = int(m.group(1))
            current_ep["n_balls"] = int(m.group(2))
            current_ep["n_delivered"] = int(m.group(3))
            continue

        # Total steps/time
        m = re.search(r'Total steps:\s*(\d+)\s+Total time:\s*([\d.]+)s', line)
        if m and current_ep is not None:
            current_ep["total_steps"] = int(m.group(1))
            current_ep["total_time"] = float(m.group(2))
            continue

    # Last episode
    if current_ep is not None:
        current_ep["per_ball"] = current_balls
        episodes.append(current_ep)

    return episodes


def main():
    parser = argparse.ArgumentParser(description="Parse PnP eval logs")
    parser.add_argument("logfile", help="Path to log file")
    parser.add_argument("--out", type=str, default=None,
                        help="Output prefix (default: <logfile>_metrics)")
    args = parser.parse_args()

    episodes = parse_log(args.logfile)
    if not episodes:
        print("No episodes found in log file.")
        sys.exit(1)

    print(f"Parsed {len(episodes)} episodes\n")

    # --- Aggregate ---
    gripped_per_ep = [e.get("n_gripped", sum(1 for b in e["per_ball"] if b["gripped"])) for e in episodes]
    delivered_per_ep = [e.get("n_delivered", sum(1 for b in e["per_ball"] if b["delivered"])) for e in episodes]
    n_balls_per_ep = [e.get("n_balls", len(e["per_ball"])) for e in episodes]
    time_per_ep = [e.get("total_time", 0.0) for e in episodes]
    steps_per_ep = [e.get("total_steps", 0) for e in episodes]

    total_gripped = sum(gripped_per_ep)
    total_delivered = sum(delivered_per_ep)
    total_balls = sum(n_balls_per_ep)
    full_deliver = sum(1 for d, n in zip(delivered_per_ep, n_balls_per_ep) if d == n)
    num_episodes = len(episodes)

    all_grip_dists = [b["grip_dist"] for e in episodes for b in e["per_ball"] if b["grip_dist"] is not None]
    all_drop_dists = [b["drop_dist"] for e in episodes for b in e["per_ball"] if b["drop_dist"] is not None]

    print(f"{'='*70}")
    print(f"PICK-AND-PLACE EVALUATION REPORT  ({num_episodes} episodes)")
    print(f"{'='*70}")

    print(f"\n--- GRIP SUCCESS ---")
    print(f"Total gripped: {total_gripped}/{total_balls} "
          f"({100*total_gripped/max(total_balls,1):.1f}%)")
    print_stats("Gripped/episode", gripped_per_ep)

    print(f"\n--- DELIVERY SUCCESS ---")
    print(f"Total delivered: {total_delivered}/{total_balls} "
          f"({100*total_delivered/max(total_balls,1):.1f}%)")
    print(f"Full deliveries (all balls): {full_deliver}/{num_episodes} "
          f"({100*full_deliver/max(num_episodes,1):.1f}%)")
    print_stats("Delivered/episode", delivered_per_ep)

    # Per-zone
    zone_delivered = {}
    zone_total = {}
    for e in episodes:
        for b in e["per_ball"]:
            zi = b.get("zone", 0)
            zone_total[zi] = zone_total.get(zi, 0) + 1
            if b.get("delivered", False):
                zone_delivered[zi] = zone_delivered.get(zi, 0) + 1
    if zone_total:
        print(f"\n--- PER-ZONE DELIVERY ---")
        for zi in sorted(zone_total):
            d = zone_delivered.get(zi, 0)
            t = zone_total[zi]
            print(f"  Zone {zi}: {d}/{t} delivered ({100*d/max(t,1):.1f}%)")

    print(f"\n--- TIMING ---")
    print_stats("Time/episode (s)", time_per_ep)
    print_stats("Steps/episode", steps_per_ep)

    if all_grip_dists:
        print(f"\n--- PER-BALL ACCURACY ---")
        print_stats("Grip distance", all_grip_dists)
        print_buckets("grip_dist", all_grip_dists, thresholds=(0.05, 0.08, 0.1, 0.2, 0.5))
    if all_drop_dists:
        print_stats("Drop distance", all_drop_dists)
        print_buckets("drop_dist", all_drop_dists, thresholds=(0.05, 0.08, 0.1, 0.2, 0.5))

    # Per-ball-index success
    max_balls = max(n_balls_per_ep) if n_balls_per_ep else 0
    if max_balls > 0:
        print(f"\n--- SUCCESS BY BALL ORDER ---")
        for bi in range(max_balls):
            grip_count = sum(1 for e in episodes if bi < len(e["per_ball"]) and e["per_ball"][bi]["gripped"])
            del_count = sum(1 for e in episodes if bi < len(e["per_ball"]) and e["per_ball"][bi]["delivered"])
            total_at_idx = sum(1 for e in episodes if bi < len(e["per_ball"]))
            print(f"  Ball {bi+1}: grip={grip_count}/{total_at_idx} "
                  f"({100*grip_count/max(total_at_idx,1):.1f}%)  "
                  f"deliver={del_count}/{total_at_idx} "
                  f"({100*del_count/max(total_at_idx,1):.1f}%)")

    print(f"{'='*70}")

    # --- Save JSON ---
    out_prefix = args.out or os.path.splitext(args.logfile)[0] + "_metrics"
    json_path = out_prefix + ".json"
    summary = {
        "num_episodes": num_episodes,
        "grip_rate": total_gripped / max(total_balls, 1),
        "delivery_rate": total_delivered / max(total_balls, 1),
        "full_delivery_rate": full_deliver / max(num_episodes, 1),
        "total_gripped": total_gripped,
        "total_delivered": total_delivered,
        "total_balls": total_balls,
        "grip_dists": all_grip_dists,
        "drop_dists": all_drop_dists,
        "gripped_per_ep": gripped_per_ep,
        "delivered_per_ep": delivered_per_ep,
        "time_per_ep": time_per_ep,
        "steps_per_ep": steps_per_ep,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nMetrics saved to: {json_path}")

    # --- Chart: single summary dashboard ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Pick-and-Place Evaluation  ({num_episodes} episodes, "
                     f"{total_balls} balls)", fontsize=14, fontweight="bold")

        # 1. Success rates (bar chart)
        ax = axes[0]
        grip_pct = 100 * total_gripped / max(total_balls, 1)
        del_pct = 100 * total_delivered / max(total_balls, 1)
        full_pct = 100 * full_deliver / max(num_episodes, 1)
        bars = ax.bar(["Grip", "Delivery", "Full Episode"],
                      [grip_pct, del_pct, full_pct],
                      color=["steelblue", "seagreen", "mediumpurple"],
                      edgecolor="white", width=0.5)
        for bar, val in zip(bars, [grip_pct, del_pct, full_pct]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Success Rates")
        mean_del = np.mean(delivered_per_ep) if delivered_per_ep else 0
        max_b = max(n_balls_per_ep) if n_balls_per_ep else 0
        ax.text(0.5, -0.12, f"Mean delivered/ep: {mean_del:.2f}/{max_b}",
                ha="center", transform=ax.transAxes, fontsize=10)

        # 2. Distance summary (grip & drop mean/p90/max)
        ax = axes[1]
        metrics_labels = []
        metrics_vals = []
        colors = []
        if all_grip_dists:
            ga = np.array(all_grip_dists)
            metrics_labels += ["Grip\nmean", "Grip\np90"]
            metrics_vals += [ga.mean(), np.percentile(ga, 90)]
            colors += ["steelblue"] * 2
        if all_drop_dists:
            da = np.array(all_drop_dists)
            metrics_labels += ["Drop\nmean", "Drop\np90"]
            metrics_vals += [da.mean(), np.percentile(da, 90)]
            colors += ["seagreen"] * 2
        if metrics_vals:
            bars = ax.bar(metrics_labels, metrics_vals, color=colors, edgecolor="white", width=0.5)
            for bar, val in zip(bars, metrics_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)
            ax.set_ylabel("Distance (m)")
            ax.set_title("Grip & Drop Distance")

        # 3. Timing summary (mean/median/p90/max)
        ax = axes[2]
        if time_per_ep:
            ta = np.array(time_per_ep)
            t_labels = ["Mean", "Median", "P90", "Max"]
            t_vals = [ta.mean(), np.median(ta), np.percentile(ta, 90), ta.max()]
            bars = ax.bar(t_labels, t_vals, color="coral", edgecolor="white", width=0.5)
            for bar, val in zip(bars, t_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}s", ha="center", va="bottom", fontsize=9)
            ax.set_ylabel("Time (s)")
            ax.set_title("Episode Time")

        plt.tight_layout()
        chart_path = out_prefix + ".png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Charts saved to: {chart_path}")
    except ImportError:
        print("matplotlib not available, skipping charts")


if __name__ == "__main__":
    main()
