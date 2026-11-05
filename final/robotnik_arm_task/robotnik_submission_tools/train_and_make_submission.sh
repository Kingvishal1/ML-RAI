#!/usr/bin/env bash
# Uses expert_demos/*.npz already recorded on this Ubuntu machine.
# Refuses training if fewer than two valid release-at-target traces exist.
set -euo pipefail
[[ -d robotnik_sensor_bc_v2 && -d robotnik_submission_tools ]] || {
    echo 'Run from robotnik_arm_task root after installation' >&2; exit 2;
}
[[ -d expert_demos ]] || { echo 'Missing expert_demos/; collect or copy actual .npz traces first.' >&2; exit 2; }
RUN_ID="$(date +%Y%m%d_%H%M%S)_${BASHPID}"
CANDIDATES="trained_expert_candidates_${RUN_ID}"
python robotnik_submission_tools/train_from_verified_expert.py \
   --data-dir expert_demos --out "$CANDIDATES" --epochs 300 --device cpu
python robotnik_submission_tools/make_submission_zip.py \
   --project "$PWD" --candidate-dir "$CANDIDATES" \
   --out "robotnik_submission_${RUN_ID}.zip"
echo "Trained candidates: $CANDIDATES"
echo "Submission: robotnik_submission_${RUN_ID}.zip"
