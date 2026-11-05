#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")" 2>/dev/null || true
# In-place project root install required. Refuse if source not present.
[[ -f robotnik_sensor_bc_v2/robotnik_v20_release_teacher_one_ball.py ]] || { echo 'Run from robotnik_arm_task root' >&2; exit 2; }
export PYTHONPATH="$PWD:$HOME/Desktop/UCL/ML/RoboEnv/simulation_and_control${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p v20_release_demos
# Changed release circle centres slightly to get distinct successful episodes.
# A valid trace needs natural detachment + ground contact + full ball in circle.
SCENARIOS=(
 '2.50 3.00 3.00 3.50'
 '2.47 3.03 3.00 3.50'
 '2.52 2.98 3.03 3.48'
 '2.48 3.02 2.97 3.52'
 '2.53 2.97 3.02 3.52'
 '2.45 3.05 2.98 3.47'
 '2.55 2.95 3.05 3.50'
 '2.49 3.01 3.00 3.55'
 '2.51 2.99 3.05 3.45'
 '2.46 3.04 2.95 3.55'
)
run_id="$(date +%Y%m%d_%H%M%S)_$$"
valid=0
for i in "${!SCENARIOS[@]}"; do
  read -r a b c d <<< "${SCENARIOS[$i]}"
  path="v20_release_demos/expert_${run_id}_${i}.npz"
  echo "=== RELEASE TEACHER TRIAL $((i+1))/${#SCENARIOS[@]} $path ==="
  # Expert is used ONLY to create imitation-learning labels. No student action is scripted.
  if python -u robotnik_sensor_bc_v2/robotnik_v20_release_teacher_one_ball.py \
      --headless --carry_controller expert --placement_controller expert \
      --base_checkpoint 060_bc_base_gelu_e70_h512_l3_16min_working.pt \
      --arm_checkpoint 051_bc_arm_gelu_e70_h512_l3_3min_working.pt \
      --grasp_checkpoint robotnik_sensor_bc_v2/grasp_sensor_residual_bc_v4.npz \
      --target "$a" "$b" 1.0 --dropoff_xy "$c" "$d" \
      --drop_zone_radius 0.50 --carry_timeout 220 --max_steps 350000 \
      --place_on_ground --ground_ball_center_z 0.55 --ground_place_speed_max 0.095 \
      --record_expert_npz "$path" 2>&1 | tee "${path%.npz}.log"; then
    echo "Teacher rollout command finished; validating physical result..."
  else
    echo "Teacher rollout command failed; trying next scenario."
  fi
  if [[ -f "$path" ]]; then
    if python - "$path" <<'PY_TEST'
import sys
sys.path.insert(0,'robotnik_submission_tools')
from train_v20_one_ball_release import read_episode
p=sys.argv[1]
try:
 x,y,h,err=read_episode(p)
 print('VERIFIED release demo:',p,'frames:',len(x),'landing error:',err)
except Exception as exc:
 print('REJECTED:',p,exc); sys.exit(1)
PY_TEST
    then valid=$((valid+1)); fi
  fi
  echo "Valid new demonstrations: $valid"
  if [[ $valid -ge 4 ]];then break;fi
done
[[ $valid -ge 4 ]] || { echo "Need 4 successful distinct demonstrations, got $valid" >&2; exit 3; }
