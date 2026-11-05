ONE BALL / 10g — LEARNED LOWERING ACTUATOR PATCH

Problem in earlier 10g code: arm reaching-policy torque was mixed with placement
joint servo control. Position gain 0.08 and 20Nm limit could track poorly.

This patch does not add any predetermined arm movement or finger-opening action.
The trained neural network still selects all lowering joint references and
when/what to release. The low-level joint position servo now:
- uses gain 0.32 rather than 0.08, force up to 35Nm instead of 20Nm,
  never above each URDF joint effort limit;
- caps each new model-predicted joint reference at 0.035 rad per 20ms;
- avoids mixing the prior reaching-policy torques with the placement servo;
- logs tracking errors as V20_LEARNED_ACTUATOR.

YOU MUST STILL RETRAIN the release student with genuine 10g corrective data.
There is no proven physical landing, and the model itself may be predicting
incorrect lateral motion. Increasing actuator force alone cannot fix a
mis-trained joint-target policy. No claim of script-free training: existing
teacher observations/labels came from an expert.

INSTALL: unzip -o V20_10G_LEARNED_ARM_TRACKING_FIX.zip in robotnik_arm_task.
No weights or teacher demos included. Existing model checkpoint required.

FAST TEST (no retraining): bash RUN_TEST_CORRECTIVE.sh
Then grep -E 'V20_LEARNED_ACTUATOR|V4_GRAVITY_HOLD_PASS|V9_GROUND_PLACE_ABORT|V20_RELEASE_STUDENT_ACTION|V20_CIRCULAR_DROP_RESULT' v20_corrective_test.log | tail -40

If XY still drifts, collect new student correction data then retrain:
bash RUN_COLLECT_STUDENT_CORRECTIONS.sh
bash RUN_RETRAIN_CORRECTIVE.sh
bash RUN_TEST_CORRECTIVE.sh

NOTE: train_v20_corrective.py currently refuses to overwrite an existing
trained_v20_corrective_release directory; rename it first, and maintain
v20_short_bend_demos as the 10g dataset.
