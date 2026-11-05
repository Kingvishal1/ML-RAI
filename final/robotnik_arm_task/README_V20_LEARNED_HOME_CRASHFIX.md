# V20 — Learned arm return crash fix (isolated)

This package fixes the reproducible `TypeError` occurring immediately after the physically verified ball release when V20 changes from `release_open` to `return_home`.

## What happened

Arm inference occurred earlier in the same timestep, so the previous code had not calculated `learned_home_world_target` when it immediately evaluated `ee_pos - learned_home_world_target`. The target was `None`.

## What changed

- Added a pure base-frame to world-frame **goal-coordinate transform**.
- Compute that goal immediately during the `release_open -> return_home` transition.
- Recompute goal each return timestep before feeding it to the existing trained arm model.
- Check and log target validity before measuring return success.

No edits to original V20 `robotnik_expert_controller.py`, gripper checkpoint, finger control, carry controller, drop-zone module, or original base/arm checkpoints. The trained arm ML policy still provides return movement, **not IK or scripted joint tracking**. Reaching its Cartesian home position does **not** guarantee all seven joints match original home angles.

## Install

From `~/Desktop/UCL/ML/ML-RAI-affan-latest/final/robotnik_arm_task`:

```bash
cp robotnik_sensor_bc_v2/robotnik_expert_controller_learned_return.py \
  robotnik_sensor_bc_v2/robotnik_expert_controller_learned_return.py.before_crashfix
unzip -o ~/Downloads/robotnik_V20_LEARNED_HOME_CRASHFIX.zip
python -m py_compile robotnik_sensor_bc_v2/robotnik_expert_controller_learned_return.py
bash RUN_V20_LEARNED_HOME_FIXED_GUI.sh 2>&1 | tee robotnik_v20_home_fixed.log
```

Expected evidence:

`V20_DROP_ZONE_LANDING ... inside_circle=True`
`V20_LEARNED_HOME_START ...`
`V20_LEARNED_HOME_GOAL world_xyz=... start_error_m=...`
`V20_LEARNED_HOME t=... ee_error_m=...`

A successful home return requires `V20_LEARNED_HOME_PASS`. If it times out, the original arm BC policy may not reach the home Cartesian target; a new trained return policy could be necessary to guarantee the desired posture. `V20_READY_NEXT_TARGET` only appears on measured return success. The GUI keeps running after that state.

This is verified by syntax and coordinate-transform unit tests only, **not a PyBullet end-to-end test**.
