#!/usr/bin/env bash
# All carry/placement proposals come from trained networks; abort if missing.
set -euo pipefail
MODEL_DIR="${PLACEMENT_MODEL_DIR:-trained_placement_bc}"
[[ -f "$MODEL_DIR/carry_candidate_bc.npz" && -f "$MODEL_DIR/placement_bc.npz" ]] || {
  echo 'Train verified placement demonstrations first with train_placement_bc.py' >&2
  exit 2
}
bash robotnik_submission_tools/run_expert.sh \
 --carry_controller bc --carry_checkpoint "$MODEL_DIR/carry_candidate_bc.npz" \
 --placement_controller bc --placement_checkpoint "$MODEL_DIR/placement_bc.npz" "$@"
