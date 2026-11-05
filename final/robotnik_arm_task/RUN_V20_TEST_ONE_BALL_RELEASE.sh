#!/usr/bin/env bash
set -euo pipefail
[[ -f trained_v20_one_ball_release/release_student.npz ]] || { echo 'Train release_student.npz first' >&2; exit 2; }
export PYTHONPATH="$PWD:$HOME/Desktop/UCL/ML/RoboEnv/simulation_and_control${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u robotnik_sensor_bc_v2/robotnik_v20_one_ball_release_student.py \
  --carry_controller expert --placement_controller bc \
  --placement_checkpoint trained_v20_one_ball_release/release_student.npz \
  --base_checkpoint 060_bc_base_gelu_e70_h512_l3_16min_working.pt \
  --arm_checkpoint 051_bc_arm_gelu_e70_h512_l3_3min_working.pt \
  --grasp_checkpoint robotnik_sensor_bc_v2/grasp_sensor_residual_bc_v4.npz \
  --target 2.50 3.00 1.00 --dropoff_xy 3.00 3.50 \
  --drop_zone_radius 0.50 --carry_timeout 220 --max_steps 350000 \
  --place_on_ground --ground_ball_center_z 0.55 --ground_place_speed_max 0.095 "$@" \
  2>&1 | tee v20_release_student_test.log
