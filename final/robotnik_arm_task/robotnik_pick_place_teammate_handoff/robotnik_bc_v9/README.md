# Robotnik V9 — Behavioural Cloning carry + release

## Exactly what this source package does

The expert controller now has an optional **ground-placement phase**: after the ball reaches the drop-off XY, the base stops, the arm follows a PyBullet IK joint target down to a guarded near-floor pose while the learned gripper keeps holding, and the fingers open only after the ball pose is stable. Ground-contact and final XY checks determine PASS. A turn-direction latch also avoids switching between +pi/-pi while the carry target is behind the base.

The separate **V9 BC student controller is not updated to execute the ground-lowering phase**. Its carry/release networks must not be used to claim ground-placement success. The supported ground-place path in this package is the expert runner; full physics verification must be done in the user's Ubuntu Robotnik environment.

| Stage | V9 student action source |
|---|---|
| Approach the ball | Existing trained base BC checkpoint (`060_...pt`) |
| Position arm | Existing trained arm BC checkpoint (`051_...pt`) |
| Close/maintain grasp | Existing V4 learned six-joint BC checkpoint (`grasp_sensor_residual_bc_v4.npz`) |
| Carry ball | **New learned carry MLP**: 5 observed features → linear speed and yaw speed (2 outputs) |
| Identify destination | **Rule-based state guard**: physical ball XY is within 0.045 m of target |
| Open gripper | **New learned release MLP**: 17 observed features → six finger position targets |
| Verify release | Physical PyBullet contact disappearance + normal-gravity falling ball |

This is **not** fully end-to-end learned control: a rule-based state machine selects tasks and checks arrival, and physics validation remains programmed. No expert steering or expert finger-opening command is used when `--carry_controller bc --release_controller bc` is selected. For a physically meaningful learned release *decision*, future training would need positive/negative release timing labels. The current release student imitates the expert's finger **actions** in the release phase, usually all zeros—this is an intentionally simple imitation problem, not evidence of learned timing.

### Ground-placement success metric

**PASS** means a physical ball is grasped and carried to world XY `(3.0,3.5)`, lowered with the gripper closed to a ball-centre height of 0.045 m, opened only after the low pose settles, detached from the robot, makes stable non-robot ground contact, and finishes within 0.05 m of the requested XY. A rollout failure must not be used as a training demonstration.

The 0.045 m ball-centre target assumes a 0.03 m sphere radius and a 0.015 m release clearance above a world-Z-zero floor. Adjust it only after checking the actual floor height in the local simulator.

## Requirements and installation

Use your existing Ubuntu `robotnik_arm_task` project and the `comp0245` environment. The ground-placement install and exact commands are in [`GROUND_PLACEMENT_START_HERE.md`](../GROUND_PLACEMENT_START_HERE.md). The installer makes a timestamped backup of the current expert controller before replacing it and preserves checkpoints and demonstration files.

The two existing base/arm checkpoints must remain in the project root:

- `060_bc_base_gelu_e70_h512_l3_16min_working.pt`
- `051_bc_arm_gelu_e70_h512_l3_3min_working.pt`

The V4 grasp checkpoint is included in this source package. Your successful expert demos (`expert_demos/expert_*.npz`) remain on Ubuntu and are not included here. Do not train new models from log text; training needs complete `.npz` observations and actions.

## 2. First verify, then collect expert demonstrations

First verify the new ground-place expert once:

```bash
bash robotnik_submission_tools/run_expert.sh 2>&1 | tee robotnik_ground_place_run.log
```

Require `V9_GROUND_PLACE_POSE_VERIFIED`, `V8_RELEASE_RESULT DETACHED_AND_FELL ground_contact=True`, and `V9_GROUND_PLACEMENT_RESULT: PASS`. The runner is headless by default to avoid GUI/X-server timeouts.

Only after that succeeds, collect a ground-place demo (default pick `(2.5,3.0,1.0)`, drop-off `(3.0,3.5)`):

```bash
bash robotnik_submission_tools/collect_expert.sh
```

The collector accepts only new `.npz` files with verified ground contact and final XY error <=0.05 m. Earlier release-at-height demos are rejected for ground-placement training. Repeat with different `--target X Y Z` pickup coordinates, keep the drop-off fixed, and count only runs with the PASS markers above.

## 3. Student BC status

The BC student still has its old at-height release flow. Do **not** run its training/evaluation path for ground-place claims until the ground-lowering state is ported and closed-loop tested in the user's simulator. Training the carry/release candidate networks alone does not teach the arm to lower the ball.

If the model cannot carry or release, the summary prints `FAIL/NOT_VERIFIED` and training **does not make a false claim** about student success. You can always run the proven expert independently:

```bash
bash robotnik_submission_tools/run_expert.sh
```

## 4. Optionally archive obsolete experimental runners

After confirming the final source and trained ZIP are backed up, preview old runners:

```bash
python robotnik_submission_tools/cleanup_old_runners.py
```

Move them to a timestamped archive folder (recommended; does not delete):

```bash
python robotnik_submission_tools/cleanup_old_runners.py --apply
```

The cleanup intentionally leaves datasets, checkpoints, your current baseline files, notes, and logs untouched; the **submission ZIP** itself already excludes obsolete experiments and logs. Do not delete any `.npz` demonstrations until after training and verification.

## Scientific limitations to disclose

- Only the **expert** was confirmed successful in user-provided Ubuntu logs; **the new V9 student has not yet been physically tested**. Your Ubuntu command above performs that test.
- The V4 grasp policy was trained on older unverified virtual-attachment expert demonstrations, but inference physically grasps a dynamic ball without a weld or attachment.
- High-level arrival detection and safe wheel stop are programmed state-machine actions. The new BC policies predict **low-level action values** during carrying and release.
- Release BC imitates a nearly constant zero-radian teacher action; do not claim it learned the *decision* to release without corresponding labels.
- Successful action cloning on a small dataset is not equivalent to reliable generalisation. Hold out episodes and evaluate closed-loop at novel start locations.
- This source package contains **no fabricated trained candidate weights**; new student models must be trained from the user’s actual `.npz` files on Ubuntu.
