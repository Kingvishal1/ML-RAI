#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Existing verified teacher episodes are training records, NEVER runtime actions.
[[ -d v20_short_bend_demos ]] || { echo "Missing successful demo records from prior RUN_TRAIN_SHORT_BEND.sh";exit 2; }
[[ -d v20_corrective_rollouts ]] || { echo "Missing student rollout records: run RUN_COLLECT_STUDENT_CORRECTIONS.sh";exit 2; }
python -u robotnik_submission_tools/train_v20_corrective.py \
  --demos v20_short_bend_demos --corrections v20_corrective_rollouts \
  --out trained_v20_corrective_release --epochs 180
printf '\nTraining complete; test with: bash RUN_TEST_CORRECTIVE.sh\n'
