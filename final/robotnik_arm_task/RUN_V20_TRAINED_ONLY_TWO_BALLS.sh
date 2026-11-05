#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")" 2>/dev/null || true
# The overlay is meant to be extracted into the user's original robotnik_arm_task root.
[[ -f 060_bc_base_gelu_e70_h512_l3_16min_working.pt ]] || { echo 'MISSING: original trained base checkpoint'; exit 2; }
[[ -f 051_bc_arm_gelu_e70_h512_l3_3min_working.pt ]] || { echo 'MISSING: original trained arm checkpoint'; exit 2; }
[[ -f robotnik_sensor_bc_v2/grasp_sensor_residual_bc_v4.npz ]] || { echo 'MISSING: original trained V4 gripper checkpoint'; exit 2; }
CARRY=${V20_CARRY_MODEL:-trained_v20_full/carry_full.npz}
PLACEMENT=${V20_PLACEMENT_MODEL:-trained_v20_full/placement_full.npz}
if [[ ! -f "$CARRY" || ! -f "$PLACEMENT" ]]; then
  echo 'ML-ONLY BLOCKED: the trained carry+arm and placement+release neural checkpoints are missing.' >&2
  echo "Need: $CARRY" >&2
  echo "Need: $PLACEMENT" >&2
  echo 'No expert/scripted fallback will be used. See README_V20_ML_ONLY.md for training inputs.' >&2
  exit 3
fi
export PYTHONPATH="$PWD:$HOME/Desktop/UCL/ML/RoboEnv/simulation_and_control${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u robotnik_sensor_bc_v2/robotnik_v20_ml_only_two_ball.py \
  --carry_controller bc --carry_checkpoint "$CARRY" \
  --placement_controller bc --placement_checkpoint "$PLACEMENT" \
  --base_checkpoint 060_bc_base_gelu_e70_h512_l3_16min_working.pt \
  --arm_checkpoint 051_bc_arm_gelu_e70_h512_l3_3min_working.pt \
  --grasp_checkpoint robotnik_sensor_bc_v2/grasp_sensor_residual_bc_v4.npz \
  --target 2.5 3.0 1.0 --next_target 1.0 3.0 1.0 \
  --dropoff_xy 3.0 3.5 --drop_zone_radius 0.50 \
  --carry_timeout 240 --max_steps 350000 --place_on_ground \
  --ground_ball_center_z 0.55 --ground_place_speed_max 0.095 \
  --second_arm_torque_limit 40 --second_center_arrival_radius 0.10 \
  "$@"
