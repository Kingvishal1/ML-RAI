#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$HOME/Desktop/UCL/ML/RoboEnv/simulation_and_control${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p v20_short_bend_demos
# Short task: original ball at approx (2.5,3.0,1.0), NEW small zone centre (2.85,3.20).
# The original (3.0,3.5) circle is NOT being used for this small-bend experiment.
# Teacher uses programmed motions only to generate labelled examples. Runtime student is neural.
SCENARIOS=(
 '2.50 3.00 2.85 3.20'
 '2.49 3.01 2.85 3.20'
 '2.52 2.98 2.85 3.20'
 '2.48 3.02 2.85 3.20'
 '2.51 2.99 2.85 3.20'
 '2.47 3.03 2.85 3.20'
 '2.53 2.97 2.85 3.20'
 '2.50 3.02 2.85 3.20'
 '2.48 3.00 2.85 3.20'
 '2.52 3.01 2.85 3.20'
)
valid=0
run_id="$(date +%Y%m%d_%H%M%S)_$$"
for i in "${!SCENARIOS[@]}"; do
  read -r a b c d <<< "${SCENARIOS[$i]}"
  path="v20_short_bend_demos/expert_${run_id}_${i}.npz"
  echo "=== SHORT BEND DEMO $((i+1))/${#SCENARIOS[@]} ==="
  python -u robotnik_sensor_bc_v2/robotnik_v20_release_teacher_one_ball.py \
    --headless --carry_controller expert --placement_controller expert \
    --base_checkpoint 060_bc_base_gelu_e70_h512_l3_16min_working.pt \
    --arm_checkpoint 051_bc_arm_gelu_e70_h512_l3_3min_working.pt \
    --grasp_checkpoint robotnik_sensor_bc_v2/grasp_sensor_residual_bc_v4.npz \
    --target "$a" "$b" 1.00 --dropoff_xy "$c" "$d" \
    --drop_zone_radius 0.32 --drop_zone_release_margin 0.05 \
    --drop_zone_approach_margin 0.04 --carry_timeout 180 --max_steps 240000 \
    --place_on_ground --ground_ball_center_z 0.50 --ground_place_speed_max 0.095 \
    --record_expert_npz "$path" 2>&1 | tee "${path%.npz}.log" || true
  if [[ -f "$path" ]]; then
    if python - "$path" <<'PY_CHECK'
import sys
sys.path.insert(0,'robotnik_submission_tools')
from train_v20_one_ball_release import read_episode
try:
    x,y,h,err=read_episode(sys.argv[1]);print('VALID',len(x),'frames landing_error',err)
except Exception as exc:
    print('REJECTED',exc);sys.exit(1)
PY_CHECK
    then valid=$((valid+1));fi
  fi
  echo "Verified demonstrations: $valid / 4"
  if (( valid >= 4 ));then break;fi
done
if (( valid < 4 ));then echo "ONLY $valid/4 VERIFIED DEMOS; REFUSING TO TRAIN";exit 3;fi
python robotnik_submission_tools/train_v20_one_ball_release.py \
  --demos v20_short_bend_demos --out trained_v20_short_bend_release --epochs 140
printf '\nTRAINING COMPLETE. Now run: bash RUN_TEST_SHORT_BEND.sh\n'
