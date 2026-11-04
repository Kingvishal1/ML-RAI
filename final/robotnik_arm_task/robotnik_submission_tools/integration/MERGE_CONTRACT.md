# Robotnik V20 integration contract (team merge)

**Reference version:** V20 circular drop zone. User reports a successful standalone V20 run, but an independently verifiable successful V20 log is not present in this package. Do not claim integrated success until a team-run evaluation passes.

## Single owner for simulation, actuation, and phases

One orchestrator owns `stepSimulation` and is the ONLY writer to wheel, arm and finger joints. Do not run two controllers in parallel writing the same joints. `robotnik_navigation_bridge.py` is a pure proposal/arbitration helper; it **does not run PyBullet**, and must be integrated with the main control loop deliberately.

## Roles

1. **Your V20 reference**: existing base and arm pretrained approach, V4 learned six-finger grasp/hold, expert physical carry, circular-zone selection/lowering, expert finger-open action, gravity-based landing checks, arm return to recorded home state.
2. **Teammates' models**: independently trained base, arm, navigation or obstacle avoidance modules. The existing V20 expert handoff does not silently install or replace their weights.
3. **Shared orchestrator**: synchronize a single world observation and call each component, enforce safety priority, apply commands once per timestep.

The current `robotnik_expert_controller.py` is a standalone reference. Its motor writes must be factored into the shared orchestrator before claiming a merged, single-simulator system. Preserve the standalone expert for regression testing.

## Safety priority and action interface

Use the navigation bridge's `CollisionDecision` where collision logic proposes an avoidance command. Arbitration precedence: safety stop > collision-avoidance proposal > carry-policy proposal. Keep limits on forward/angular velocity, and do not let avoidance own finger joint commands. During RELEASE, stop the base and give one component exclusive finger ownership.

## V20 area-based placement

`--dropoff_xy` specifies the **zone centre** (default [3.0, 3.5]), not a required point target. `--drop_zone_radius` is the physical radius (default **0.50 m**). V20's release gate is conservative (default release radius 0.37 m); success requires gravity-driven detachment and stable *non-robot ground contact* with the entire ball inside the circle (ball centre within radius minus ball radius, default 0.47 m). The algorithm must use measured ball position, not merely base centre. Keep the `V20_CIRCULAR_DROP_RESULT`, `V8_RELEASE_RESULT`, `V9_GROUND_PLACEMENT_RESULT`, and `V19_ARM_RETURN_HOME` logs as acceptance signals.

The release height defaults to 0.55 m ball centre above the ground. The rendered circle has no collision body and cannot support the ball. The expert selects and reaches a safe point inside it; this is not learned zone selection. Run separately from any student's learned placement controller until that student's action and phase interfaces are updated and evaluated.

## Model paths / BC state

- Base BC approach checkpoint: `060_bc_base_gelu_e70_h512_l3_16min_working.pt` (external, not bundled).
- Arm BC approach checkpoint: `051_bc_arm_gelu_e70_h512_l3_3min_working.pt` (external, not bundled).
- V4 finger checkpoint: `robotnik_sensor_bc_v2/grasp_sensor_residual_bc_v4.npz` (bundled, hash in `MODEL_STATUS.json`).
- Optional candidate carry/release/placement networks are **not** included as validated trained weights; the legacy student path differs from the working V20 expert placement flow.

## Merge test matrix

- Run standalone V20 with `comp0245` and save complete log, checkpoint SHA-256 hashes, and success/failure summaries.
- Run base/arm replacements independently and verify phase handoff and grasp before changing the carrying code.
- Integrate obstacle avoidance into one wheel-command arbitration mechanism and test stop and detour/recovery, including stale sensor data.
- Confirm the gripper physically holds a dynamic ball under gravity; no virtual attachment/weld/antigravity assistance.
- Verify a gravity-driven release **inside the full circle**, stable floor contact, and subsequent arm-home return.
- Re-run with multiple start locations/seeds and report failures; a single success is not reliable generalisation.

**Keep:** team checkpoints, `.npz` recordings, logs demonstrating success/failure, the older verified baseline for rollback. **Do not commit:** caches, generated logs, stale archive ZIPs, entire virtual environments, unreviewed training outputs. Never run `git clean -fdx` on this project.
