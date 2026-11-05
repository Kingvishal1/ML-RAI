V20 STEP 4: ISOLATED RETURN-HOME MOTOR SERVO FIX

This patch changes only the return_home arm actuator path and return-home diagnostics.
It does NOT modify V4 grasp, release checkpoint, model predictions, finger release
angles, circle/drop checks, or learned lowering actions.

Current code uses a dedicated PyBullet position servo for lowering/release_open,
but return_home switched to a torque path. This patch makes return_home use the
same dedicated position servo; disables conflicting arm torque in that state;
and logs the per-joint return error vector.

NOTE: Full six-finger opening is NOT fixed in this patch, because the trained
network only predicts partial opening. This requires separate dataset/model
improvement if it must remain learned, rather than a scripted fallback.

Usage (from project root):
  cp robotnik_sensor_bc_v2/robotnik_v20_one_ball_release_student.py \
    robotnik_sensor_bc_v2/robotnik_v20_one_ball_release_student.py.pre_step4
  unzip -o ~/Downloads/V20_STEP4_RETURN_HOME_FIX.zip
  bash RUN_TEST_STEP4_RETURN_HOME.sh
  grep -E 'V19_ARM_RETURN|V20_RELEASE_STUDENT_ACTION|V20_CIRCULAR_DROP_RESULT' \
    v20_step4_return_home_test.log | tail -40

Validation performed: Python compile, shell syntax, diff review. No local
PyBullet closed-loop run has been performed. Verify results from your GUI log.
