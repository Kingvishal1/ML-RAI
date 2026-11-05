# V20: NO EXPERT ACTIONS AT RUNTIME — ML-only experimental replacement

## What is implemented

The code retains V20's trained base, arm and V4 six-finger grasp network. Between each of two ball deliveries it starts a new task/controller episode with no world, base, or arm pose reset. The **physical balls** share the same 0.50 m radius circular destination at `(3.0,3.5)`.

During transport, a **trained 25->9 carry model** predicts the base's `(linear,angular)` velocity AND all seven arm joint torques. That replaces BOTH the expert steering and the previous scripted arm hold. During lowering, a **trained 37->14 placement model** predicts seven arm joint target changes, six release finger targets, and a release probability. The opening command is the *latched NN prediction*, never a hardcoded expert-open action. During grasp and until opening, the unchanged V4 trained gripper controls the fingers.

No expert navigation, Cartesian lowering planner, or immediate-open fallback acts in this mode. The controller refuses to run when either replacement checkpoint is missing. The base/arm motor interfaces, kinematic measurements, task-phase transitions and physical safety checks remain conventional computer code; **"no scripts" here means no hand-coded task-movement or release commands**, not an absence of ordinary software. The lowering target offsets are tracked by a conventional low-level PD torque interface, just as motor controllers generally track policy targets.

## Current evidence / limitations — read before running

**NOT A FINISHED, TESTED 2/2-DELIVERY SYSTEM.** The original user-supplied project bundles contain the V4 gripper weights and the interface to base/arm models, but **NO verified human-demonstration datasets and NO trained 25->9 carry or 37->14 placement weights**. Log text is not a substitute for time-aligned sensor/action training arrays. This bundle deliberately includes **no invented or placeholder checkpoints**. No fully learned second delivery can be claimed yet.

## Files to install

Extract `V20_LEARNED_ONLY_RUNTIME.zip` over a copy of the existing `robotnik_arm_task` directory. This adds a distinct controller `robotnik_sensor_bc_v2/robotnik_v20_ml_only_two_ball.py` and **does not overwrite the working original V20**.

## Obtain verified HUMAN TELEOP demonstrations (default: no scripted teacher)

At least **five independent physically successful demonstration NPZ episodes** are required (four training, one held out). Record time-aligned 25-dimensional carry sensor vectors and 9-dimensional carry actions (two base speeds + seven arm torques), plus 37-dimensional placement sensor vectors and 14-dimensional placement actions (seven arm joint offsets, six finger targets, binary release decision). The corresponding feature generator functions are in `robotnik_learned_task_policy.py` and `placement_policy.py`. The recorder/teleoperation UI to produce these from a human operator is **not included**; existing V20 log files cannot reconstruct all required labels. Scripted V20 expert recordings are **not accepted** by this trainer. This avoids training on scripted expert motor commands; use verified operator teleoperation recordings. Alternatively a separately trained reinforcement-learning policy could be developed, but that training system is not included.

Every NPZ episode must include:

- `source='HUMAN_TELEOP'` (or another value containing TELEOP)
- `carry_full_obs_25` shape `(T,25)`, `carry_full_actions_9` `(T,9)`
- `placement_obs_37` shape `(U,37)`, `placement_actions_14` `(U,14)`
- `final_placement_verified=True`, `ground_contact_verified=True`, `ball_released_verified=True`, `no_virtual_attachment=True`
- `final_ball_xy_error_m` floating and `drop_zone_radius_m=0.50`; final whole ball must fit the circle

Actual physical validation matters. A human demonstration must be a real time-aligned recording from the simulation, not randomly generated labels or V20 terminal text.

## Train after real data exists

```bash
conda activate comp0245
cd ~/Desktop/UCL/ML/ML-RAI-affan-latest/final/robotnik_arm_task
python robotnik_submission_tools/train_v20_no_expert_runtime.py \
  --demos verified_teleop_demos --output trained_v20_full --epochs 300
```

The resulting `trained_v20_full/carry_full.npz` and `trained_v20_full/placement_full.npz` are candidate models only. Check `train_report.json` for held-out whole-episode results; that is **not** a closed-loop score. Collect data covering both short carry and the long second-ball carry, turns and different arm/grasp configurations.

## Run only after both trained models exist

```bash
bash RUN_V20_TRAINED_ONLY_TWO_BALLS.sh 2>&1 | tee v20_learned_only.log
```

If models are missing, this command **exits with code 3**; it does not secretly activate expert motion.

## What success means

Two `V20_MULTI_DELIVERY_PASS` events and `V20_MULTI_FINAL delivered_at_release=2/2 still_inside_and_supported=2/2`, with both objects remaining physically inside the same circle after the full simulation. No fake attachment, object teleport, arm reset or expert release. **Not demonstrated in this build.**
