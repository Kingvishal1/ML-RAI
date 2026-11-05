V20 one-ball 10g learned release gate diagnostic.
Changes ONLY: releases gate diagnostics (every 0.5s) and speed safety gate
in the new diagnostic launcher from 0.095 to 0.18 m/s.
No scripted arm/finger targets or overrides are added. Release still requires
student network probability >=0.5, 2 finger contact groups, and XY within
original circle's safe release radius. Model checkpoint remains unchanged.
The 0.18m/s gate is a diagnostic experiment and could cause a less accurate
landing than the original gate. It does not guarantee a release or success.
Run in project folder: bash RUN_DIAGNOSE_LEARNED_RELEASE.sh
Look for V20_RELEASE_GATE, V20_RELEASE_STUDENT_ACTION, V20_CIRCULAR_DROP_RESULT.
The simulator still has a programmed task state machine and the existing
expert carry option; local_drop bypasses long-distance carry motion.
