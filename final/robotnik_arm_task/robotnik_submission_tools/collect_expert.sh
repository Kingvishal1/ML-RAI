#!/usr/bin/env bash
# Saves only a demonstration that passes ground-contact and final-XY validation.
set -euo pipefail
[[ -d robotnik_sensor_bc_v2 ]] || { echo 'Run from robotnik_arm_task root' >&2; exit 2; }
mkdir -p expert_demos
ID="$(date +%Y%m%d_%H%M%S)_${BASHPID}"
NPZ="expert_demos/expert_${ID}.npz"; LOG="expert_demos/expert_${ID}.log"
bash robotnik_submission_tools/run_expert.sh --headless --record_expert_npz "$NPZ" "$@" 2>&1 | tee "$LOG"
python - "$NPZ" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'robotnik_submission_tools')
from train_from_verified_expert import read_episode
path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit('Expert trace missing; cannot count run')
result = read_episode(path)
if result is None:
    raise SystemExit('Expert demonstration NOT valid; do not train on this episode')
print('VERIFIED GROUND-PLACEMENT EXPERT DEMO:',path)
PY
