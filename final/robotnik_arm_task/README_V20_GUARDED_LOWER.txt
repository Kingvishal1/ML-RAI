V20 second-ball guarded lowering experiment (2026-11-04)

Evidence: prior run physically carried ball 2 into same circle, but grip dropped
from 2 groups to 1 just after entry into V18 ground lowering. At transition
local gripper-frame displacement was ~44mm (85mm emergency cutoff).

What changed ONLY on ball 2:
- A short stationary pre-lowering check. Keep the ORIGINAL trained V4 finger
  policy active and gripper motor force unchanged, with existing arm hold.
- Require >=2 contact groups with nonzero physical normal force and low ball
  speed before beginning lowering, otherwise abort safely (no false success).
- Lower with same V18 expert measured-ball Cartesian planner, but smaller
  per-joint targets, slower descent, and smaller XYZ steps.
- More diagnostics every 0.1s during lowering.

Unchanged: ball 1, base/arm ML checkpoints, second arm torque handover,
V4 gripper network and force, pickup, circle, expert carry & finger-open release,
physical ball gravity, two-group/contact/ground-landing verification.
No attachments, joints reset, scripted arm return, fake success.

This code is syntax-checked and planner invariants unit-checked, but PyBullet
full-run is NOT verified in assistant environment.

INSTALL: unzip into ORIGINAL V20 task folder, not previous versions.
RUN: bash RUN_V20_TWO_BALLS_GUARDED_LOWER.sh 2>&1 | tee robotnik_v20_guarded_lower.log
DIAG: grep -E "V20_SECOND_PRELOWER|V20_SECOND_GENTLE_LOWER|V20_SECOND_LOWER_CONTACT|V20_MULTI_DELIVERY_PASS|V20_MULTI_FINAL|FAIL|ABORT" robotnik_v20_guarded_lower.log | tail -70
