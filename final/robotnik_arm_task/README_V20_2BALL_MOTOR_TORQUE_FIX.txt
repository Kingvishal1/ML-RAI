V20 SECOND-BALL MOTOR-MODE / LEARNED TORQUE DIAGNOSTIC

This patch is NOT a new grasp controller. It keeps original V20 V4 learned
six-finger inference, contact checks, carry, circle release, the same two balls
and the same PyBullet simulation. It does not add scripted joint lifting, joint
teleports, IK or a separate home movement.

Cause investigated: V20 lowering uses p.POSITION_CONTROL and the next episode
uses learned torque. The previous learned policy requested ~36 Nm at joint 2,
but the runner clipped the request to 20 Nm. The second ball stalled at 0.469m.

Changes for BALL TWO ONLY:
  1. Disable residual POSITION_CONTROL motor state with zero-force VELOCITY_CONTROL
     before the second episode. No movement is applied by this operation.
  2. Allow learned torque outputs up to 40 Nm, bounded by each joint URDF effort
     rating. This is an actuator limit, not a commanded/scripted trajectory.
  3. Report joint limits, rated efforts, actual requested and clamped torques,
     so a new stall can be diagnosed without another blind patch.

The source is syntax checked but the integrated PyBullet simulation has not
been executed here. Never count ball 2 unless real contact, gravity hold and
ball-in-circle checks all pass. If the arm still stalls, the pretrained policy
may need training on post-placement states, or there may be an arm constraint
or collision. Don't raise force limits beyond URDF ratings.

INSTALL: unzip into ~/Desktop/UCL/ML/ML-RAI-affan-latest/final/robotnik_arm_task
RUN: bash RUN_V20_TWO_BALLS_MOTOR_TORQUE_FIX.sh 2>&1 | tee robotnik_v20_2ball_torque_fix.log
CHECK: grep -E 'V20_SECOND_ARM_MOTOR_UNLOCK|V20_SECOND_ARM_LIMITS|V20_SECOND_ARM_DIAG|V20_MULTI_ARM_PROGRESS|V20_MULTI_DELIVERY_PASS|V20_MULTI_FINAL|STALLED' robotnik_v20_2ball_torque_fix.log | tail -70
