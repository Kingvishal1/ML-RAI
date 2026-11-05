# Robotnik pick, carry and ground-place — teammate handoff

**Purpose:** Share *your* Robotnik control component with a teammate working on LiDAR collision avoidance. This is NOT an integrated obstacle-avoidance application yet.

## What is included

- `robotnik_bc_v9/robotnik_expert_controller.py`: modified expert controller with a 180-degree turn-direction latch and a guarded arm-lowering phase before release. **The new ground-placement path has not yet been run in the user's Ubuntu simulator.**
- `robotnik_bc_v9/robotnik_bc_controller.py`: student runner for separately trained carry/release BC models. Source only: **new student weights have not been trained or physics-verified here.**
- `robotnik_bc_v9/grasp_sensor_residual_bc_v4.npz`: existing pretrained 17-to-6 grasp checkpoint.
- `robotnik_bc_v9/sensor_bc_v4.py`, `candidate_bc_policy.py`: policy implementations.
- `robotnik_bc_v9/train_from_verified_expert.py` and scripts: optional student training/evaluation.
- `integration/robotnik_navigation_bridge.py`: pure-python proposal arbitration, no PyBullet calls.
- `integration/MERGE_CONTRACT.md`: commands, sensor interface, state machine and acceptance checks.
- `integration/test_bridge.py`: small unit tests for priority and wheel conversion.
- `build_complete_zip_on_ubuntu.py`: repack on Ubuntu with your **real** `.pt` checkpoints and optional actual trained student weights.

Follow [GROUND_PLACEMENT_START_HERE.md](GROUND_PLACEMENT_START_HERE.md) to install and run the new expert path. It must pass once before collecting ground-placement demonstrations.

## Important missing artifacts

This source handoff DOES NOT contain:

- `060_bc_base_gelu_e70_h512_l3_16min_working.pt` (base checkpoint).
- `051_bc_arm_gelu_e70_h512_l3_3min_working.pt` (arm checkpoint).
- `carry_candidate_bc.npz` and `release_candidate_bc.npz` (new student weights, if not trained yet).
- LiDAR obstacle sensor processing (belongs to the teammate).

To make a **complete teammate ZIP containing your actual base/arm checkpoints**, extract this package on your Ubuntu computer and run from the project root:

```bash
cd ~/Desktop/UCL/ML/ML-RAI-affan-latest/final/robotnik_arm_task
python ~/Downloads/robotnik_pick_place_teammate_handoff/build_complete_zip_on_ubuntu.py "$PWD"
```

If you already trained genuine BC carry/release candidates and want to include them:

```bash
python ~/Downloads/robotnik_pick_place_teammate_handoff/build_complete_zip_on_ubuntu.py \
  "$PWD" --student-dir "$PWD/trained_bc_v9_YOUR_RUN"
```

This creates a separate ZIP inside `robotnik_arm_task/` ready to send. It does **not** delete or modify any source file or checkpoint. It aborts rather than omit required pretrained checkpoints.

## For teammate: integration contract (LiDAR)

1. **One simulator and one motor-command writer.** The current `robotnik_bc_controller.py` is a *standalone* runner and must not be launched simultaneously with your collision runner.
2. Import `integration/robotnik_navigation_bridge.py` into your shared main loop. Your LiDAR collision module should output `CollisionDecision` with either an avoidance `(linear_mps, angular_radps)` proposal, `stop=True`, or `active=False` when clear.
3. Calculate a BC carry proposal from the existing **5-feature carry observation** (only if a real carry BC model was trained); call `arbitrate_carry(bc_command, lidar_decision)` before turning it into wheel commands. Priority: emergency stop > LiDAR avoidance > BC carry.
4. Only the **shared main loop** writes wheels, fingers and steps PyBullet. Keep `GRASP/HOLD` finger control (V4 trained grasp) separate from `RELEASE` finger control (BC release student if verified, otherwise the working expert).
5. Expert ground placement stops at ball XY tolerance, lowers the ball near floor height, opens the fingers, and requires stable floor contact plus final XY error <=0.05 m for ground-placement PASS. The separate BC student has not yet been updated for the lowering phase.
6. First run the successful short-distance task **without** obstacles; next activate LiDAR veto; finally test avoidance detours and recovery. The current BC carry policy is *not* trained on obstacle-detour data.

## Quick source checks (requires NumPy)

```bash
python -m unittest discover -s integration -p 'test_*.py'
python -m py_compile robotnik_bc_v9/robotnik_bc_controller.py \
  robotnik_bc_v9/robotnik_expert_controller.py
```

## Running standalone before integration

Existing COMP0245 simulation with RoboEnv is required. To copy the runners into a local project, from `robotnik_arm_task`:

```bash
bash /path/to/robotnik_pick_place_teammate_handoff/robotnik_bc_v9/install.sh "$PWD"
bash robotnik_submission_tools/run_expert.sh
```

The install script backs up an existing expert controller when necessary. Prefer **not** installing directly on your teammate's shared source tree until integration is agreed. You can also run the main runner manually with your existing environment.

## Project ownership

- **Your side:** base/arm approach, V4 ML closing/hold, BC carry/release experiments, expert reference, physical grasp/release validation, navigation proposal bridge.
- **Teammate:** LiDAR obstacle detection and safe navigation proposal.
- **Shared:** motor arbitration, one simulator tick, combined end-to-end testing.

Model status and provenance: see `robotnik_bc_v9/MODEL_STATUS.json`. No student weights are invented or passed off as trained.
