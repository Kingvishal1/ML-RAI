# Ground placement: exact Ubuntu steps

This update changes the expert reference controller. It keeps the learned approach and gripper policy, adds a guarded arm-lowering phase at the drop-off, and opens only after the held ball has reached a stable near-floor pose. The first rollout is a required local simulation check; this source builder cannot run the user's RoboEnv/PyBullet setup.

## 1. Install the updated source

From Ubuntu, with the source ZIP saved in `~/Downloads`:

```bash
cd ~/Desktop/UCL/ML/ML-RAI-affan-latest/final/robotnik_arm_task
conda activate comp0245
mkdir -p ~/Downloads/robotnik_ground_place_v2
unzip -oq ~/Downloads/robotnik_pick_place_teammate_handoff_GROUND_PLACE_v2.zip -d ~/Downloads/robotnik_ground_place_v2
bash ~/Downloads/robotnik_ground_place_v2/robotnik_pick_place_teammate_handoff/robotnik_bc_v9/install.sh "$PWD"
```

The installer backs up the current expert controller before replacing it. It leaves checkpoints and existing demonstration files in place.

## 2. Run one headless expert rollout

```bash
bash robotnik_submission_tools/run_expert.sh 2>&1 | tee robotnik_ground_place_run.log
```

The runner includes `--headless` to avoid the X-server timeout observed in the earlier GUI run. Look for all of these:

```text
V9_GROUND_PLACE_START
V9_GROUND_PLACE_POSE_VERIFIED
V8_RELEASE_RESULT DETACHED_AND_FELL ... ground_contact=True
V9_GROUND_PLACEMENT_RESULT: PASS
V8_PICK_CARRY_DROP_RESULT: PASS
```

The default floor assumption is world Z=0, ball radius 0.03 m, and ball-centre placement at Z=0.045 m. If the IK solution is unreachable, the ball slips, or the ball does not settle within 0.05 m of drop-off XY, the run aborts or reports FAIL and keeps the gripper closed when it can.

## 3. Collect only verified ground-placement demonstrations

After one rollout passes, collect one demo at a time. Keep drop-off fixed and vary the pickup target:

```bash
bash robotnik_submission_tools/collect_expert.sh --target 2.45 3.05 1.0 --dropoff_xy 3.0 3.5
bash robotnik_submission_tools/collect_expert.sh --target 2.55 3.05 1.0 --dropoff_xy 3.0 3.5
bash robotnik_submission_tools/collect_expert.sh --target 2.45 2.95 1.0 --dropoff_xy 3.0 3.5
bash robotnik_submission_tools/collect_expert.sh --target 2.55 2.95 1.0 --dropoff_xy 3.0 3.5
bash robotnik_submission_tools/collect_expert.sh --target 2.50 3.10 1.0 --dropoff_xy 3.0 3.5
```

Each command saves an `.npz` trace only if the physical ground-contact and final-XY checks pass. The training validator rejects old at-height release traces. **Do not run the BC student training/evaluation command for ground-placement claims yet:** its student runner still needs this lowering phase ported and tested.

## 4. If a run fails

Keep the full log. Do not treat it as a successful demo. Send the final summary and the first `V9_GROUND_*` or `V8_CARRY_FAIL` line so the IK target, joint order, or clearance can be corrected from evidence.
