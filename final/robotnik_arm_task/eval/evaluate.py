#!/usr/bin/env python3
"""Evaluate BC policy accuracy across multiple random targets.

Reports separate metrics for base navigation and arm reaching phases,
plus overall end-to-end accuracy.

Usage:
  # Split policy (single target)
  python evaluate.py --base_checkpoint base.pt --arm_checkpoint arm.pt --num_targets 20 --headless

  # Multi-drop-zone pick-and-place
  python evaluate.py --pick_and_place --multi_target --num_episodes 10 --headless \
      --base_checkpoint base.pt --arm_checkpoint arm.pt

  # Selector-only (no simulation)
  python evaluate.py --selector_only --selector_checkpoint sel.pt --num_episodes 50
"""

import argparse
import json
import os
import sys
import time as _time

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

if "--headless" in sys.argv:
    os.environ["ROBOENV_HEADLESS"] = "1"

import numpy as np

from run_policy_sim import (
    run_sim, load_policy, load_selector,
    run_pick_and_place_sim,
    REACH_TOLERANCE,
)
from robotnik_arm_task import (
    compute_base_goal, DROP_ZONES_DEFAULT, NUM_DROP_ZONES,
    shortest_path_expert, build_selector_scene,
    generate_random_drop_zones,
)


def generate_targets(n, seed=50):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-5, 5, n)
    ys = rng.uniform(-5, 5, n)
    zs = rng.uniform(0.5, 1.0, n)
    return [np.array([xs[i], ys[i], zs[i]]) for i in range(n)]


def print_stats(label, values):
    if not values:
        print(f"  {label:20s}: no data")
        return
    arr = np.array(values)
    print(f"  {label:20s}: mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}  std={arr.std():.4f}")


def print_buckets(label, values, thresholds=(0.05, 0.1, 0.2, 0.5, 1.0)):
    if not values:
        return
    n = len(values)
    for t in thresholds:
        count = sum(1 for v in values if v < t)
        print(f"    {label} < {t}m: {count}/{n} ({100*count/n:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC policy across multiple targets")
    parser.add_argument("--base_checkpoint", type=str, default=None)
    parser.add_argument("--arm_checkpoint", type=str, default=None)
    parser.add_argument("--base_threshold", type=float, default=None)
    parser.add_argument("--num_targets", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=100000,
                        help="Max sim steps (default: 100000 ≈ 100s at 1ms timestep)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: random)")
    parser.add_argument("--base_only", action="store_true",
                        help="Only evaluate base phase (stop at phase switch)")
    parser.add_argument("--arm_only", action="store_true",
                        help="Only evaluate arm phase (use expert base)")
    # Multi-target / pick-and-place evaluation
    parser.add_argument("--multi_target", action="store_true",
                        help="Evaluate multi-target episodes")
    parser.add_argument("--num_targets_per_episode", type=int, default=3,
                        help="Targets per multi-target episode (default: 3)")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of episodes to evaluate (default: 10)")
    parser.add_argument("--selector_checkpoint", type=str, default=None,
                        help="Path to selector checkpoint (omit for expert)")
    parser.add_argument("--selector_only", action="store_true",
                        help="Evaluate selector decisions only (no simulation)")
    parser.add_argument("--per_target_max_steps", type=int, default=100000,
                        help="Max steps per target in multi-target mode (default: 100000)")
    parser.add_argument("--heading_threshold", type=float, default=0.15)
    parser.add_argument("--arm_damping", type=float, default=0.0)
    parser.add_argument("--pick_and_place", action="store_true",
                        help="Evaluate pick-and-place mode (multi-drop-zone)")
    parser.add_argument("--xy_range", type=float, default=5.0,
                        help="XY workspace half-extent for target/zone generation (default: 5.0)")
    parser.add_argument("--multi_target_seed", type=int, default=52,
                        help="Random seed for target generation (default: 42)")
    parser.add_argument("--expert", action="store_true",
                        help="Use expert controllers instead of learned policies (no checkpoints needed)")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = int(np.random.default_rng().integers(0, 2**31))
    print(f"Seed: {args.seed}")

    # Determine mode
    split_mode = args.base_checkpoint is not None and args.arm_checkpoint is not None
    single_mode = False
    base_only = args.base_only
    arm_only = args.arm_only

    if args.expert:
        # Expert mode doesn't need checkpoints; set split_mode so the plumbing works
        split_mode = True
        print("Mode: EXPERT controllers (no learned policies)")

    if base_only and not args.base_checkpoint:
        parser.error("--base_only requires --base_checkpoint")
    if arm_only and not args.arm_checkpoint:
        parser.error("--arm_only requires --arm_checkpoint")

    if base_only:
        split_mode = True
        single_mode = False
        if not args.arm_checkpoint:
            args.arm_checkpoint = None
    if arm_only:
        split_mode = True
        single_mode = False
        if not args.base_checkpoint:
            args.base_checkpoint = None

    if not split_mode and not args.selector_only and not args.expert:
        parser.error("Specify --base_checkpoint and --arm_checkpoint (or use --expert)")

    # Load policies
    import torch
    if args.mps and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    policy_kwargs = {}

    if split_mode and not args.expert:
        if args.base_checkpoint:
            base_model, base_mean, base_std, base_lm, base_ls, base_fnames = \
                load_policy(args.base_checkpoint, device=device)
            policy_kwargs.update(base_model=base_model, base_mean=base_mean, base_std=base_std,
                                 base_label_mean=base_lm, base_label_std=base_ls,
                                 base_feature_names=base_fnames)
            print(f"Base policy: {args.base_checkpoint}")

        if args.arm_checkpoint:
            arm_model, arm_mean, arm_std, arm_lm, arm_ls, arm_fnames = \
                load_policy(args.arm_checkpoint, device=device)
            policy_kwargs.update(arm_model=arm_model, arm_mean=arm_mean, arm_std=arm_std,
                                 arm_label_mean=arm_lm, arm_label_std=arm_ls,
                                 arm_feature_names=arm_fnames)
            print(f"Arm policy: {args.arm_checkpoint}")

        print(f"device={device}  switch={'xy<'+str(args.base_threshold)+'m' if args.base_threshold else 'settle'}")

    # --- Selector-only evaluation (no simulation) ---
    if args.selector_only:
        from run_policy_sim import predict_selector

        if not args.selector_checkpoint:
            parser.error("--selector_only requires --selector_checkpoint")

        sel_model, sel_mean, sel_std = load_selector(args.selector_checkpoint, device=device)
        print(f"Selector: {args.selector_checkpoint}  device={device}")

        rng = np.random.default_rng(args.seed)
        n_per_ep = args.num_targets_per_episode
        xy_r = args.xy_range
        print(f"Mode: multi-drop-zone (randomized zones per episode, xy_range={xy_r})")

        total_decisions = 0
        correct_decisions = 0
        expert_total_dist = 0.0
        model_total_dist = 0.0

        for ep in range(args.num_episodes):
            targets = np.array([[rng.uniform(-xy_r, xy_r), rng.uniform(-xy_r, xy_r),
                                 rng.uniform(0.5, 1.0)]
                                for _ in range(n_per_ep)])

            robot_pos = np.array([0.0, 0.0, 0.0])
            robot_bearing = 0.0
            remaining = set(range(n_per_ep))

            mdz_zones = generate_random_drop_zones(rng, xy_range=xy_r)
            ball_zone_assignments = [int(rng.integers(0, len(mdz_zones)))
                                     for _ in range(n_per_ep)]

            ep_model_dist = 0.0

            while len(remaining) > 1:
                expert_idx, _ = shortest_path_expert(
                    robot_pos, targets, remaining, mdz_zones, ball_zone_assignments)
                scene_obs, slot_order = build_selector_scene(
                    robot_pos, robot_bearing, targets, remaining,
                    mdz_zones, ball_zone_assignments)

                model_idx = predict_selector(sel_model, sel_mean, sel_std,
                                             scene_obs, slot_order, device)

                if model_idx == expert_idx:
                    correct_decisions += 1
                total_decisions += 1

                # Compute leg distance for model's choice
                ep_model_dist += np.linalg.norm(robot_pos[:2] - targets[model_idx, :2])
                dz = mdz_zones[ball_zone_assignments[model_idx]]
                ep_model_dist += np.linalg.norm(targets[model_idx, :2] - dz[:2])

                # After delivery, robot is at the assigned drop zone
                remaining.remove(model_idx)
                robot_pos = mdz_zones[ball_zone_assignments[model_idx]].copy()
                robot_bearing = rng.uniform(-np.pi, np.pi)

            # Last remaining target (no decision needed, just add distance)
            if remaining:
                last = remaining.pop()
                ep_model_dist += np.linalg.norm(robot_pos[:2] - targets[last, :2])
                dz = mdz_zones[ball_zone_assignments[last]]
                ep_model_dist += np.linalg.norm(targets[last, :2] - dz[:2])

            # Compute expert total distance (full optimal path from origin)
            _, full_order = shortest_path_expert(
                np.array([0, 0, 0]), targets, set(range(n_per_ep)),
                mdz_zones, ball_zone_assignments)

            ep_expert_dist_full = 0.0
            pos = np.array([0.0, 0.0, 0.0])
            for oi, idx in enumerate(full_order):
                ep_expert_dist_full += np.linalg.norm(pos[:2] - targets[idx, :2])
                dz = mdz_zones[ball_zone_assignments[idx]]
                ep_expert_dist_full += np.linalg.norm(targets[idx, :2] - dz[:2])
                pos = dz.copy()

            expert_total_dist += ep_expert_dist_full
            model_total_dist += ep_model_dist

            if (ep + 1) % max(1, args.num_episodes // 10) == 0 or ep == 0:
                print(f"  Episode {ep+1}/{args.num_episodes}  "
                      f"model_dist={ep_model_dist:.2f}  expert_dist={ep_expert_dist_full:.2f}")

        acc = 100.0 * correct_decisions / max(total_decisions, 1)
        avg_expert = expert_total_dist / args.num_episodes
        avg_model = model_total_dist / args.num_episodes
        overhead = 100.0 * (avg_model - avg_expert) / max(avg_expert, 1e-6)

        print(f"\n{'='*60}")
        print(f"SELECTOR-ONLY EVALUATION  ({args.num_episodes} episodes, "
              f"{n_per_ep} targets each)")
        print(f"{'='*60}")
        print(f"Decision accuracy:     {correct_decisions}/{total_decisions} ({acc:.1f}%)")
        print(f"Avg expert distance:   {avg_expert:.2f}")
        print(f"Avg model distance:    {avg_model:.2f}")
        print(f"Distance overhead:     {overhead:+.1f}%")
        print(f"{'='*60}")
        return

    # --- Multi-target evaluation ---
    if args.multi_target:
        sel_model = sel_mean = sel_std = None
        use_expert_selector = args.selector_checkpoint is None
        if args.selector_checkpoint:
            sel_model, sel_mean, sel_std = load_selector(args.selector_checkpoint, device=device)
            print(f"Selector: {args.selector_checkpoint}")
        else:
            print("Selector: expert (nearest-neighbor)")

        rng = np.random.default_rng(args.seed)
        all_results = []

        mode_label = "pick-and-place" if args.pick_and_place else "multi-target"
        print(f"\nEvaluating {args.num_episodes} {mode_label} episodes "
              f"({args.num_targets_per_episode} targets each, seed={args.seed})\n")

        for ep in range(args.num_episodes):
            n = args.num_targets_per_episode
            MIN_TARGET_DIST = 1.5  # m, minimum XY distance from origin
            targets = []
            for _ in range(n):
                while True:
                    t = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0.5, 1.0)])
                    if np.linalg.norm(t[:2]) >= MIN_TARGET_DIST:
                        break
                targets.append(t)
            print(f"\n[Episode {ep+1}/{args.num_episodes}]")

            mdz_zones = generate_random_drop_zones(rng)
            ball_zone_assignments = [int(rng.integers(0, len(mdz_zones))) for _ in range(n)]
            result = run_pick_and_place_sim(
                args, targets, device=device,
                split_mode=split_mode, single_mode=single_mode,
                drop_zones=mdz_zones,
                ball_zone_assignments=ball_zone_assignments,
                sel_model=sel_model, sel_mean=sel_mean, sel_std=sel_std,
                use_expert_selector=use_expert_selector,
                **policy_kwargs,
            )
            all_results.append(result)

        # --- Aggregate report ---
        if args.pick_and_place:
            print(f"\n{'='*70}")
            print(f"{mode_label.upper()} EVALUATION REPORT  ({args.num_episodes} episodes, "
                  f"{args.num_targets_per_episode} balls each)")
            print(f"{'='*70}")

            gripped_per_ep = [r.get("balls_gripped", 0) for r in all_results]
            delivered_per_ep = [r.get("balls_delivered", 0) for r in all_results]
            n_balls_per_ep = [r.get("n_balls", 0) for r in all_results]
            time_per_ep = [r.get("total_time", 0.0) for r in all_results]
            steps_per_ep = [r.get("total_steps", 0) for r in all_results]

            total_gripped = sum(gripped_per_ep)
            total_delivered = sum(delivered_per_ep)
            total_balls = sum(n_balls_per_ep)
            full_deliver = sum(1 for g, n in zip(delivered_per_ep, n_balls_per_ep) if g == n)

            print(f"\n--- GRIP SUCCESS ---")
            print(f"Total gripped: {total_gripped}/{total_balls} "
                  f"({100*total_gripped/max(total_balls,1):.1f}%)")
            print_stats("Gripped/episode", gripped_per_ep)

            print(f"\n--- DELIVERY SUCCESS ---")
            print(f"Total delivered: {total_delivered}/{total_balls} "
                  f"({100*total_delivered/max(total_balls,1):.1f}%)")
            print(f"Full deliveries (all balls): {full_deliver}/{args.num_episodes} "
                  f"({100*full_deliver/max(args.num_episodes,1):.1f}%)")
            print_stats("Delivered/episode", delivered_per_ep)

            # Per-zone delivery counts
            zone_delivered = {}
            zone_total = {}
            for r in all_results:
                for pb in r.get("per_ball", []):
                    zi = pb.get("drop_zone_idx", 0)
                    zone_total[zi] = zone_total.get(zi, 0) + 1
                    if pb.get("delivered", False):
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

            # Per-ball breakdown
            all_grip_dists = []
            all_drop_dists = []
            for r in all_results:
                for pb in r.get("per_ball", []):
                    if "grip_dist" in pb:
                        all_grip_dists.append(pb["grip_dist"])
                    if "drop_dist" in pb:
                        all_drop_dists.append(pb["drop_dist"])

            if all_grip_dists:
                print(f"\n--- PER-BALL ACCURACY ---")
                print_stats("Grip distance", all_grip_dists)
                print_buckets("grip_dist", all_grip_dists,
                              thresholds=(0.05, 0.08, 0.1, 0.2, 0.5))
            if all_drop_dists:
                print_stats("Drop distance", all_drop_dists)
                print_buckets("drop_dist", all_drop_dists,
                              thresholds=(0.05, 0.08, 0.1, 0.2, 0.5))

            # Distance overhead vs optimal (selector evaluation)
            if not use_expert_selector:
                print(f"\n--- SELECTOR QUALITY ---")
                print(f"(Compare visit order distance vs optimal multi-drop TSP)")

            print(f"{'='*70}")

            # --- Save metrics to JSON ---
            timestamp = _time.strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(os.path.dirname(__file__), "results")
            os.makedirs(out_dir, exist_ok=True)
            json_path = os.path.join(out_dir, f"pnp_eval_{timestamp}.json")

            summary = {
                "timestamp": timestamp,
                "seed": args.seed,
                "num_episodes": args.num_episodes,
                "num_targets_per_episode": args.num_targets_per_episode,
                "base_checkpoint": args.base_checkpoint,
                "arm_checkpoint": args.arm_checkpoint,
                "expert": args.expert,
                "grip_rate": total_gripped / max(total_balls, 1),
                "delivery_rate": total_delivered / max(total_balls, 1),
                "full_delivery_rate": full_deliver / max(args.num_episodes, 1),
                "total_gripped": total_gripped,
                "total_delivered": total_delivered,
                "total_balls": total_balls,
                "grip_dists": all_grip_dists,
                "drop_dists": all_drop_dists,
                "gripped_per_ep": gripped_per_ep,
                "delivered_per_ep": delivered_per_ep,
                "time_per_ep": time_per_ep,
                "steps_per_ep": steps_per_ep,
                "per_episode": [],
            }
            for ep_i, r in enumerate(all_results):
                summary["per_episode"].append({
                    "episode": ep_i,
                    "balls_gripped": r.get("balls_gripped", 0),
                    "balls_delivered": r.get("balls_delivered", 0),
                    "n_balls": r.get("n_balls", 0),
                    "total_time": r.get("total_time", 0.0),
                    "total_steps": r.get("total_steps", 0),
                    "visit_order": r.get("visit_order", []),
                    "per_ball": r.get("per_ball", []),
                })

            with open(json_path, "w") as f:
                json.dump(summary, f, indent=2, default=float)
            print(f"\nMetrics saved to: {json_path}")

            # --- Generate chart ---
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, 3, figsize=(16, 5))
                fig.suptitle(f"Pick-and-Place Evaluation  ({args.num_episodes} episodes, "
                             f"{total_balls} balls)", fontsize=14, fontweight="bold")

                # 1. Success rates
                ax = axes[0]
                grip_pct = 100 * total_gripped / max(total_balls, 1)
                del_pct = 100 * total_delivered / max(total_balls, 1)
                full_pct = 100 * full_deliver / max(args.num_episodes, 1)
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
                ax.text(0.5, -0.12,
                        f"Mean delivered/ep: {mean_del:.2f}/{args.num_targets_per_episode}",
                        ha="center", transform=ax.transAxes, fontsize=10)

                # 2. Distance summary (grip & drop mean/p90/max)
                ax = axes[1]
                labels, vals, colors = [], [], []
                if all_grip_dists:
                    ga = np.array(all_grip_dists)
                    labels += ["Grip\nmean", "Grip\np90"]
                    vals += [ga.mean(), np.percentile(ga, 90)]
                    colors += ["steelblue"] * 2
                if all_drop_dists:
                    da = np.array(all_drop_dists)
                    labels += ["Drop\nmean", "Drop\np90"]
                    vals += [da.mean(), np.percentile(da, 90)]
                    colors += ["seagreen"] * 2
                if vals:
                    bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.5)
                    for bar, val in zip(bars, vals):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
                    ax.set_ylabel("Distance (m)")
                    ax.set_title("Grip & Drop Distance")

                # 3. Timing summary
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
                chart_path = os.path.join(out_dir, f"pnp_eval_{timestamp}.png")
                plt.savefig(chart_path, dpi=150)
                plt.close()
                print(f"Charts saved to: {chart_path}")
            except ImportError:
                print("matplotlib not available, skipping charts")

            return

        print(f"{'='*70}")
        return

    # Generate targets
    targets = generate_targets(args.num_targets, seed=args.seed)
    print(f"\nEvaluating {len(targets)} targets (seed={args.seed})\n")

    # For base_only: reduce max_steps so it doesn't run forever after switch
    if base_only:
        args.max_steps = min(args.max_steps, 100000)

    # Run evaluations
    results = []
    for ti, target in enumerate(targets):
        print(f"[{ti+1}/{len(targets)}] Target: {np.round(target, 3)}", end="  ")
        sys.stdout.flush()

        result = run_sim(args, target, device=device,
                         split_mode=split_mode, single_mode=single_mode,
                         **policy_kwargs)
        results.append(result)

        # Compute expert standoff goal for this target
        goal_xy, goal_bearing = compute_base_goal(target)
        result["standoff_xy"] = goal_xy.tolist()
        result["standoff_bearing"] = float(goal_bearing)

        # Base XY distance to standoff at best EE moment
        if result["base_xy_at_best"] is not None:
            bxy = np.array(result["base_xy_at_best"])
            result["base_xy_err"] = float(np.linalg.norm(bxy - goal_xy))
            bearing_err = result["base_bearing_at_best"] - goal_bearing
            result["base_bearing_err"] = float(np.abs(np.arctan2(np.sin(bearing_err), np.cos(bearing_err))))
        else:
            result["base_xy_err"] = None
            result["base_bearing_err"] = None

        # One-line summary
        status = "OK" if result["ok"] else "CRASH"
        xy_str = f"  base_xy_err={result['base_xy_err']:.4f}" if result["base_xy_err"] is not None else ""
        if result["phase_switched"]:
            print(f"{status}  base_best_ee={result['base_best_ee']:.4f}{xy_str}  "
                  f"arm_best={result['arm_best_ee']:.4f}  "
                  f"final={result['final_err']:.4f}")
        else:
            print(f"{status}  best={result['best_err']:.4f}{xy_str}  final={result['final_err']:.4f}")

    # ===== Aggregate Report =====
    ok_results = [r for r in results if r["ok"]]
    crashed = len(results) - len(ok_results)
    n = len(results)

    print(f"\n{'='*70}")
    print(f"EVALUATION REPORT  ({n} targets)")
    print(f"{'='*70}")
    print(f"Completed: {len(ok_results)}/{n}  Crashed: {crashed}/{n}")

    if not ok_results:
        print("All runs crashed.")
        return

    # --- Overall ---
    print(f"\n--- OVERALL ---")
    reached = sum(1 for r in ok_results if r["reached"])
    print(f"Reached target ({REACH_TOLERANCE*1000:.0f}mm): {reached}/{len(ok_results)} "
          f"({100*reached/len(ok_results):.1f}%)")
    print_stats("Final EE error", [r["final_err"] for r in ok_results])
    print_stats("Best EE error", [r["best_err"] for r in ok_results])
    print_buckets("best", [r["best_err"] for r in ok_results])

    # --- Base phase ---
    base_results = [r for r in ok_results if r["base_best_ee"] < np.inf]
    if base_results:
        print(f"\n--- BASE PHASE ({len(base_results)} runs) ---")
        switched = [r for r in base_results if r["phase_switched"]]
        print(f"Phase switched: {len(switched)}/{len(base_results)}")
        if switched:
            print_stats("Switch time (s)", [r["switch_time"] for r in switched])
        print_stats("Best EE at base", [r["base_best_ee"] for r in base_results])
        print_stats("Final EE at base", [r["base_final_ee"] for r in base_results])
        print_buckets("base_best_ee", [r["base_best_ee"] for r in base_results])

        # XY distance to expert standoff position
        xy_errs = [r["base_xy_err"] for r in base_results if r["base_xy_err"] is not None]
        bearing_errs = [r["base_bearing_err"] for r in base_results if r["base_bearing_err"] is not None]
        if xy_errs:
            print(f"\n  Accuracy vs expert standoff position:")
            print_stats("XY dist to standoff", xy_errs)
            print_buckets("xy_err", xy_errs, thresholds=(0.05, 0.1, 0.2, 0.5, 1.0))
        if bearing_errs:
            bearing_degs = [np.degrees(e) for e in bearing_errs]
            print_stats("Bearing error (deg)", bearing_degs)
            print_buckets("bearing_err_deg", bearing_degs, thresholds=(5, 10, 20, 45, 90))

    # --- Arm phase ---
    arm_results = [r for r in ok_results if r["arm_best_ee"] < np.inf]
    if arm_results:
        print(f"\n--- ARM PHASE ({len(arm_results)} runs) ---")
        print_stats("Best EE at arm", [r["arm_best_ee"] for r in arm_results])
        print_stats("Final EE at arm", [r["arm_final_ee"] for r in arm_results])
        print_buckets("arm_best", [r["arm_best_ee"] for r in arm_results])
        # Improvement from base->arm
        improvements = []
        for r in arm_results:
            if r["base_final_ee"] < np.inf:
                improvements.append(r["base_final_ee"] - r["arm_best_ee"])
        if improvements:
            print_stats("Arm improvement", improvements)
            improved = sum(1 for v in improvements if v > 0)
            print(f"  Arm improved over base: {improved}/{len(improvements)} "
                  f"({100*improved/len(improvements):.1f}%)")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
