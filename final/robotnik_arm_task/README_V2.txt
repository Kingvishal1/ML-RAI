V20 ONE BALL 10g - release retraining V2

What this changes:
- Training-only counterfactual labels now include student-visited states inside 10cm
  of the circle center, at ball heights 0.32..0.60m, below 0.18m/s.
- Arm-target and release commands at TEST time still come ONLY from trained network.
- Increase the release decision learning weight, keep original neural architecture.
- Retain all existing trained checkpoints. Output folder is timestamped.
- Existing task phase changes, low-level actuator servo, physics and safety gates
  remain programmed; local_drop avoids long expert carry movement.

This is imitation-learning / counterfactual supervised training, NOT reinforcement
learning or a script-free training method. Safe-release labels are geometric
heuristics used ONLY for training; closed-loop simulation success is not guaranteed.
Data generated at 50g must not be mixed with 10g data without disclosure.

RUN:
  bash RUN_COLLECT_RELEASE_V2.sh
  bash RUN_TRAIN_RELEASE_V2.sh
  bash RUN_TEST_RELEASE_V2.sh

AUDIT:
  grep -E 'RELEASE_POSITIVE_FRAMES|TRAINED|V20_RELEASE_GATE|V20_RELEASE_STUDENT_ACTION|V20_CIRCULAR_DROP_RESULT|ABORT' v20_release_v2_*.log
