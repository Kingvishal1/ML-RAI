#!/usr/bin/env bash
# Installation is non-destructive: never delete checkpoints/training recordings.
set -euo pipefail
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${1:-$PWD}"
[[ -d "$PROJECT/robotnik_sensor_bc_v2" ]] || { echo "Not a Robotnik project: $PROJECT" >&2; exit 2; }
mkdir -p "$PROJECT/robotnik_submission_tools"
# Preserve the original working expert if its current bytes differ.
ORIGINAL="$PROJECT/robotnik_sensor_bc_v2/robotnik_expert_controller.py"
if [[ -f "$ORIGINAL" ]] && ! cmp -s "$SOURCE/robotnik_expert_controller.py" "$ORIGINAL"; then
  BACKUP="$PROJECT/robotnik_sensor_bc_v2/robotnik_expert_controller_before_bc_v9_$(date +%Y%m%d_%H%M%S).py"
  cp "$ORIGINAL" "$BACKUP"
  echo "Expert backed up: $BACKUP"
fi
for name in robotnik_expert_controller.py robotnik_bc_controller.py candidate_bc_policy.py \
            sensor_bc_v4.py grasp_sensor_residual_bc_v4.npz; do
  cp "$SOURCE/$name" "$PROJECT/robotnik_sensor_bc_v2/$name"
done
for name in train_from_verified_expert.py check_candidate_policy.py \
            cleanup_old_runners.py make_submission_zip.py \
            run_student_bc.sh run_expert.sh collect_expert.sh train_and_evaluate.sh \
            README.md MODEL_STATUS.json; do
  cp "$SOURCE/$name" "$PROJECT/robotnik_submission_tools/$name"
done
chmod +x "$PROJECT/robotnik_submission_tools/"*.sh
python -m py_compile "$PROJECT/robotnik_sensor_bc_v2/robotnik_expert_controller.py" \
                        "$PROJECT/robotnik_sensor_bc_v2/robotnik_bc_controller.py" \
                        "$PROJECT/robotnik_sensor_bc_v2/candidate_bc_policy.py"
echo 'Ground-place expert source installed. Full Robotnik/PyBullet rollout is still required.'
echo 'Original pretrained base/arm files and expert demos were not deleted.'
