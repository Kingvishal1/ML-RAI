#!/usr/bin/env bash
# Train real expert demonstrations, then TEST the learned students in physics.
# A zip is generated even on physics failure, marked explicitly unverified.
set -euo pipefail
[[ -d robotnik_sensor_bc_v2 && -d robotnik_submission_tools ]] || {
  echo 'Run from robotnik_arm_task root after install.sh' >&2; exit 2; }
[[ -d expert_demos ]] || { echo 'No expert_demos/ folder; collect valid NPZ demonstrations first.' >&2; exit 2; }
ID="$(date +%Y%m%d_%H%M%S)_${BASHPID}"
CANDIDATES="trained_bc_v9_${ID}"
python robotnik_submission_tools/train_from_verified_expert.py \
  --data-dir expert_demos --out "$CANDIDATES" --epochs 300 --device cpu
EVAL_JSON="$CANDIDATES/student_closed_loop.json"
EVAL_LOG="$CANDIDATES/student_closed_loop.log"
if bash robotnik_submission_tools/run_student_bc.sh "$CANDIDATES" --result_json "$EVAL_JSON" 2>&1 | tee "$EVAL_LOG"; then
  echo 'Student simulation finished. Checking physics evidence in JSON...'
else
  echo 'Student simulation crashed/failed to start; student not verified.'
fi
OUT="robotnik_bc_v9_submission_${ID}.zip"
ZIP_ARGS=(--project "$PWD" --candidate-dir "$CANDIDATES" --out "$OUT")
if [[ -f "$EVAL_JSON" ]]; then ZIP_ARGS+=(--evaluation-json "$EVAL_JSON"); fi
python robotnik_submission_tools/make_submission_zip.py "${ZIP_ARGS[@]}"
python - "$EVAL_JSON" "$OUT" <<'PY'
import json,sys
from pathlib import Path
report=Path(sys.argv[1]);archive=Path(sys.argv[2])
validated=False
if report.is_file():
    d=json.loads(report.read_text())
    validated=bool(d.get('task_success') and d.get('full_closed_loop_student_test')
                   and d.get('carry_mode')=='bc' and d.get('release_mode')=='bc')
print('COMPLETE ARCHIVE:',archive)
print('LEARNED STUDENT SIMULATION:', 'PASS' if validated else 'NOT VERIFIED / FAIL')
print('EXPERT RUNNER AND PRETRAINED CHECKPOINTS ARE ALWAYS INCLUDED IN ARCHIVE.')
PY
