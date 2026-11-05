#!/usr/bin/env bash
# Each invocation creates a new original expert trace and physics log.
set -euo pipefail
[[ -d robotnik_sensor_bc_v2 ]] || { echo 'Run from robotnik_arm_task project root' >&2; exit 2; }
mkdir -p expert_demos
RUN_ID="$(date +%Y%m%d_%H%M%S)_${BASHPID}"
NPZ="expert_demos/expert_${RUN_ID}.npz"
LOG="expert_demos/expert_${RUN_ID}.log"
# Use the same runner and optional command-line overrides as the working simulation.
# A later training stage independently checks whether this was a physical success.
set -o pipefail
bash robotnik_submission_tools/run_working_expert.sh --record_expert_npz "$NPZ" "$@" 2>&1 | tee "$LOG"
python - "$NPZ" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,'robotnik_submission_tools')
from train_from_verified_expert import read_episode
path=Path(sys.argv[1])
if not path.exists(): raise SystemExit('No expert NPZ produced')
result=read_episode(path)
if result is None:
    raise SystemExit('FAIL: physics did not verify release at target; demonstration excluded from training.')
print('PASS: physically verified release at target. Landing position is not the success criterion.')
print('Saved expert trace:',path)
PY
