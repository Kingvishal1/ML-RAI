V20: sequential two-ball pick and place within ONE PyBullet world
==============================================================

Install into your ORIGINAL V20 folder:
  ~/Desktop/UCL/ML/ML-RAI-affan-latest/final/robotnik_arm_task
Unzip this package there. It adds NEW files; it does not overwrite the original
robotnik_sensor_bc_v2/robotnik_expert_controller.py or model checkpoints.

Run:
  bash RUN_V20_TWO_BALLS_GUI.sh 2>&1 | tee robotnik_v20_two_ball.log

First pickup ball:  (2.5, 3.0, 1.0) — red
Second pickup ball: (1.0, 3.0, 1.0) — orange
Both release at world XY (3.0, 3.5), radius 0.50 m.
Second ball is created in the same physics world at the start. Its mass is zero
until a verified two-finger-group grasp, just like original V20. The robot,
base, arm, floor, and first delivered ball are NOT reset between cycles.

Important: the V20 neural policies for base approach, arm approach, and V4
finger closure are unchanged. The original V20 expert carry and placement
logic remains unchanged. The only new sequencing logic starts the next cycle
after physical delivery to the circular zone is verified. NO arm-home step or
IK home-return routine runs in the two-ball variant. No artificial attachment.

If ball 1 fails physical delivery, ball 2 is not started. A failed delivery
is never counted as successful. Balls remain physically simulated.

Useful log lines:
  grep -E 'V20_MULTI_|V4_GRAVITY_HOLD_PASS|V20_DROP_ZONE_LANDING|V20_CIRCULAR_DROP_RESULT|Traceback|FAIL' robotnik_v20_two_ball.log

After testing 2 balls, you can add another pickup with another
  --next_target X Y Z
argument to the script; both (or all) will use the same --dropoff_xy circle.

Testing note: This package passed Python syntax and source-integrity checks,
but has not run against the user's full RoboEnv / PyBullet simulation here.
Important unknown: original V20 base approach may not reliably reach ball 2
from the end of the first placement while the arm remains lowered. If it
fails, collect its phase and contact logs before changing the learned policy.

Undo: remove RUN_V20_TWO_BALLS_GUI.sh and the new
robotnik_sensor_bc_v2/robotnik_expert_controller_two_ball.py. Original unchanged.
