V20 original learned gripper/release preserved. EXPERIMENTAL target-input correction for second ball.

Logs prove latest learned continuation was installed but stalled when the second arm target was capped to z≈0.683 m although the actual second ball was at z=1.0 m. This changes ONLY the target fed to the existing trained arm policy during second-ball approach, from a capped 0.24m vector to the actual full ball goal. No motion script/IK/joint servo is added. The same V20 grasp/carry/release logic remains in this file. The new diagnostics print the model torque and joint positions to check whether the model is physically able to recover.

INSTALL (run from the ORIGINAL affan-latest final/robotnik_arm_task):
  unzip -o PATH/robotnik_V20_SECOND_BALL_FULL_GOAL_DIAGNOSTIC.zip
  python -m py_compile robotnik_sensor_bc_v2/robotnik_expert_controller_two_ball_full_goal.py
  bash RUN_V20_TWO_BALLS_FULL_GOAL.sh 2>&1 | tee robotnik_v20_full_goal.log
  grep -E 'V20_MULTI_DELIVERY_PASS|V20_MULTI_ARM_PROGRESS|V20_SECOND_ARM_DIAG|V20_MULTI_ARM_STALLED|V20_MULTI_FINAL' robotnik_v20_full_goal.log | tail -50

Expected confirmation on second ball: V20_SECOND_ARM_DIAG goal=[1.0, 3.0, 1.0] approximately (the final pocket correction may slightly shift XY in proximity). If q/EE remain flat and learned_tau barely changes, current BC arm likely lacks recovery behaviour from ground-placement joint pose. We must train a learned recovery/transition policy from actual post-placement initial states, not add scripted joint control. No PyBullet execution was possible for this patch; syntax verified only.
