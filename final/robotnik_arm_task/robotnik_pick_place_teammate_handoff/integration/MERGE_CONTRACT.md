# Robotnik integration contract — BC grasp/carry/release + collision avoidance

**Status:** Merge-ready interface and unit tests, **not yet tested in the teammate's integrated PyBullet simulation**. Preserve the working expert as a regression baseline. Keep existing ML checkpoint files unchanged.

## Golden rule: ONE owner for simulation and motors

One `main.py` must own `p.stepSimulation()`, `p.setJointMotorControl2()` / `setJointMotorControlArray()`, time stepping, and final wheel commands. Do **not** launch your teammate's simulator and `robotnik_bc_controller.py` simultaneously. The historical V9 runner is standalone and must be refactored at the state/command boundaries to merge properly. In the integrated design, `robotnik_navigation_bridge.py` computes actions but **never touches PyBullet**.

## Responsibility split

- **Your component:** pretrained base+arm approach, V4 BC finger grasp/hold, new BC carry policy, new BC release policy, grasp/release monitoring and ball-goal success criterion.
- **Teammate's component:** obstacle sensing, collision-risk decision, local avoidance proposal; should not set wheel motors directly or command gripper joints.
- **Shared orchestrator:** samples *one* synchronized simulator state, calls both proposals, resolves priority, applies one set of wheel and finger commands, then steps simulation once.

## Typed contract for collision avoidance

`CollisionDecision(active=False)` means proceed with the carry policy's proposed command. When `active=True`, the teammate provides a safe `(linear_mps, angular_radps)` proposal. `stop=True` is a hard veto that forces both to zero. Never bypass the safety arbiter; **safety stop > collision avoidance > learned carry**. All speed values are in m/s and rad/s. Negative linear commands can be permitted for avoidance with appropriate limits.

```
obs = bc_carry_observation(ball_xy, dropoff_xy, carry_origin_xy,
                           base_yaw, base_velocity_xy, base_yaw_rate)
student = carry_action(carry_policy, obs, max_linear_mps=0.025,
                       max_angular_radps=0.4)
avoid = teammate_avoidance(sensor_snapshot)  # must return CollisionDecision
final = arbitrate_carry(student, avoid, 0.025, 0.4)
wheel_commands = wheel_speeds(final, WHEEL_RADIUS, WHEEL_BASE_WIDTH)
# ONLY SHARED ORCHESTRATOR writes wheel motors; ONLY ONE simulator step afterwards.
```

In `APPROACH`, route the base BC approach command through the same collision gate using a separate speed limit suited to approach. In `CARRY`, route the 5→2 BC output through it. In `RELEASE`, force base motion to zero and give exclusive control of the six fingers to the release policy. Obstacle avoidance **does not** control gripper targets. If collision logic halts the robot during `CARRY`, pause the route and monitor grip rather than falsely declaring arrival.

## High-level state machine

`APPROACH -> ALIGN -> GRASP -> GRAVITY_HOLD -> CARRY -> RELEASE -> VERIFY -> DONE`, with `ABORT` if grip fails or timeout. Only the shared state machine changes phases. `arrived_at_goal()` is evaluated **only** after successful grasp and in the CARRY phase; arrival is based on **ball XY**, not base centre. When the ball is within 0.045 m of `(3.0, 3.5)`, the robot commands zero wheel velocity, deactivates V4 hold inference and uses the release BC policy (17→6 outputs), one writer per actuator group.

## BC observation and action formats — do not change silently

`carry_candidate_bc.npz`: `[remaining_m, lateral_path_error_m, heading_error_rad, base_forward_mps, base_yaw_rate_radps]` → `[linear_mps, angular_radps]`. The path origin is the **ball world XY when carry begins**. `release_candidate_bc.npz`: 17 features computed by `sensor_bc_v4.sensor_features()` (ball relative to EE 3; finger joint positions 6; finger joint velocities 6; base-frame linear velocity 2) → 6 finger target positions **in radians**. The original pretrained V4 policy remains responsible for closing/holding; release BC is activated only in the release phase.

**Important collision-avoidance limitation:** The carry policy has no obstacle information and was trained on obstacle-free expert demonstrations. Teammate's avoidance must safely override it. After a detour, the original five-feature policy uses the fixed origin-to-goal straight line as reference; this may make recovery difficult. Test detour/rejoin behavior and retrain with obstacle-aware expert demonstrations if needed; don't claim avoidance is learned by BC.

## Merge acceptance tests

1. Baseline run: standalone expert picks up, carries 0.71 m, and opens with gravity-driven ball fall (final floor XY is informational only).
2. Student-alone: genuine `carry_candidate_bc.npz` and `release_candidate_bc.npz` perform the same physical task (not merely low training MSE).
3. Collision-free integration: same results in teammate's single-simulator orchestrator.
4. Obstacle integration: avoidance overrides carry when blocked; one wheel writer, no collision, no premature release, no grip loss; route resumes when clear.
5. Emergency stop and stale/invalid sensor handling: safe zero wheel command.
6. Artifacts: source, checkpoint hashes, dependency versions, demonstration provenance, and actual test logs; disclose any student-only failures.

## Training, cleanup, final submission

Train on the genuine successful `expert_demos/expert_*.npz` recordings on Ubuntu with existing V9 `train_and_evaluate.sh`, but **do not automatically treat a failed student test as submission-ready**. Include model checkpoints in the final merge only after recording their hashes. Keep one stable expert baseline for debugging. Archive old runner versions only after combined integration passes; do not delete teammate files, training recordings, original checkpoints, or reproducibility logs.
