#!/usr/bin/env python3
"""Evaluate BC policy accuracy across multiple random targets.

Reports separate metrics for base navigation and arm reaching phases,
plus overall end-to-end accuracy.

Usage:
  # Single policy
  python evaluate.py --checkpoint model.pt --num_targets 20 --headless

  # Split policy
  python evaluate.py --base_checkpoint base.pt --arm_checkpoint arm.pt --num_targets 20 --headless

  # Base only (skip arm phase)
  python evaluate.py --base_checkpoint base.pt --num_targets 20 --base_only --headless

  # Arm only (skip base, start near target)
  python evaluate.py --arm_checkpoint arm.pt --num_targets 20 --arm_only --headless
"""

import argparse
import os
import sys

if "--headless" in sys.argv:
    os.environ["ROBOENV_HEADLESS"] = "1"

import numpy as np

from run_policy_sim import (
    run_sim, load_policy, load_gravity_comp, load_selector,
    run_multi_target_sim, REACH_TOLERANCE,
)
from robotnik_arm_task import compute_base_goal


def generate_targets(n, seed=42):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-5, 5, n)
    ys = rng.uniform(-5, 5, n)
    zs = rng.uniform(0.2, 1.0, n)
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
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--base_checkpoint", type=str, default=None)
    parser.add_argument("--arm_checkpoint", type=str, default=None)
    parser.add_argument("--base_threshold", type=float, default=None)
    parser.add_argument("--num_targets", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=100000,
                        help="Max sim steps (default: 100000 ≈ 100s at 1ms timestep)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_only", action="store_true",
                        help="Only evaluate base phase (stop at phase switch)")
    parser.add_argument("--arm_only", action="store_true",
                        help="Only evaluate arm phase (use expert base)")
    parser.add_argument("--gravity_comp", type=str, default=None,
                        help="Path to gravity compensation network checkpoint")
    # Multi-target evaluation
    parser.add_argument("--multi_target", action="store_true",
                        help="Evaluate multi-target episodes")
    parser.add_argument("--num_targets_per_episode", type=int, default=3,
                        help="Targets per multi-target episode (default: 3)")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of multi-target episodes to evaluate (default: 10)")
    parser.add_argument("--selector_checkpoint", type=str, default=None,
                        help="Path to selector checkpoint (omit for expert nearest-neighbor)")
    parser.add_argument("--per_target_max_steps", type=int, default=100000,
                        help="Max steps per target in multi-target mode (default: 100000)")
    parser.add_argument("--heading_threshold", type=float, default=0.15)
    parser.add_argument("--arm_damping", type=float, default=0.0)
    args = parser.parse_args()

    # Determine mode
    split_mode = args.base_checkpoint is not None and args.arm_checkpoint is not None
    single_mode = args.checkpoint is not None
    base_only = args.base_only
    arm_only = args.arm_only

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

    if not split_mode and not single_mode:
        parser.error("Specify --checkpoint or --base_checkpoint/--arm_checkpoint")

    # Load policies
    import torch
    if args.mps and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    policy_kwargs = dict(device=device)

    if single_mode:
        model, mean, std, label_mean, label_std, _ = load_policy(args.checkpoint, device=device)
        policy_kwargs.update(model=model, mean=mean, std=std,
                             label_mean=label_mean, label_std=label_std)
        print(f"Single policy: {args.checkpoint}  device={device}")
    else:
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

    if args.gravity_comp:
        grav_model, grav_q_mean, grav_q_std, grav_tau_mean, grav_tau_std = \
            load_gravity_comp(args.gravity_comp, device=device)
        print(f"Gravity comp: {args.gravity_comp}")
        policy_kwargs.update(grav_model=grav_model, grav_q_mean=grav_q_mean,
                             grav_q_std=grav_q_std, grav_tau_mean=grav_tau_mean,
                             grav_tau_std=grav_tau_std)

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

        print(f"\nEvaluating {args.num_episodes} multi-target episodes "
              f"({args.num_targets_per_episode} targets each, seed={args.seed})\n")

        for ep in range(args.num_episodes):
            n = args.num_targets_per_episode
            targets = [np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0.2, 1.0)])
                        for _ in range(n)]
            print(f"\n[Episode {ep+1}/{args.num_episodes}]")

            result = run_multi_target_sim(
                args, targets,
                split_mode=split_mode, single_mode=single_mode,
                sel_model=sel_model, sel_mean=sel_mean, sel_std=sel_std,
                use_expert_selector=use_expert_selector,
                **policy_kwargs,
            )
            all_results.append(result)

        # --- Aggregate multi-target report ---
        print(f"\n{'='*70}")
        print(f"MULTI-TARGET EVALUATION REPORT  ({args.num_episodes} episodes, "
              f"{args.num_targets_per_episode} targets each)")
        print(f"{'='*70}")

        reached_per_ep = [r["targets_reached"] for r in all_results]
        total_per_ep = [r["n_targets"] for r in all_results]
        time_per_ep = [r["total_time"] for r in all_results]
        steps_per_ep = [r["total_steps"] for r in all_results]

        total_reached = sum(reached_per_ep)
        total_targets = sum(total_per_ep)
        full_clear = sum(1 for r, t in zip(reached_per_ep, total_per_ep) if r == t)

        print(f"\n--- TARGETS ---")
        print(f"Total reached: {total_reached}/{total_targets} "
              f"({100*total_reached/total_targets:.1f}%)")
        print(f"Full clears (all targets): {full_clear}/{args.num_episodes} "
              f"({100*full_clear/args.num_episodes:.1f}%)")
        print_stats("Reached/episode", reached_per_ep)
        print_stats("Time/episode (s)", time_per_ep)
        print_stats("Steps/episode", steps_per_ep)

        # Per-target error stats
        all_final_errs = []
        all_best_errs = []
        for r in all_results:
            for pt in r["per_target"]:
                all_final_errs.append(pt.get("final_err", float("inf")))
                all_best_errs.append(pt.get("best_err", float("inf")))

        print(f"\n--- PER-TARGET ACCURACY ---")
        print_stats("Final EE error", all_final_errs)
        print_stats("Best EE error", all_best_errs)
        print_buckets("best_err", all_best_errs)

        # Visit ordering quality vs expert
        if not use_expert_selector:
            from robotnik_arm_task import nearest_neighbor_expert
            order_matches = 0
            order_total = 0
            for ep_i, result in enumerate(all_results):
                # Reconstruct what expert would have chosen
                n = result["n_targets"]
                per_target = result["per_target"]
                # We can't perfectly reconstruct expert order without replaying,
                # but we can compare the visit_order length
                order_total += 1
            if order_total > 0:
                print(f"\n--- ORDERING ---")
                print(f"(Ordering comparison requires replay; "
                      f"compare with --selector_checkpoint=None for expert baseline)")

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

        result = run_sim(args, target, split_mode=split_mode, single_mode=single_mode,
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
