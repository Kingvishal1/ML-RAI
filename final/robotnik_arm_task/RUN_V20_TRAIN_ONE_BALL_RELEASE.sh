#!/usr/bin/env bash
set -euo pipefail
[[ -f robotnik_submission_tools/train_v20_one_ball_release.py ]] || { echo 'Run from robotnik_arm_task root' >&2; exit 2; }
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -u robotnik_submission_tools/train_v20_one_ball_release.py \
  --demos v20_release_demos --out trained_v20_one_ball_release \
  --epochs "${RELEASE_EPOCHS:-140}" 2>&1 | tee v20_release_training.log
python VERIFY_V20_ONE_BALL_RELEASE.py --checkpoint trained_v20_one_ball_release/release_student.npz
