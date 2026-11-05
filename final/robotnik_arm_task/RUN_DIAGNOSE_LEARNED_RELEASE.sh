#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ -s trained_v20_corrective_release/release_student.npz ]] || { echo 'Missing learned release checkpoint' >&2; exit 2; }
export PYTHONPATH="$PWD:$HOME/Desktop/UCL/ML/RoboEnv/simulation_and_control${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u robotnik_sensor_bc_v2/robotnik_v20_one_ball_release_student.py \
  --carry_controller expert --placement_controller bc \
  --placement_checkpoint trained_v20_corrective_release/release_student.npz \
  --base_checkpoint 060_bc_base_gelu_e70_h512_l3_16min_working.pt \
  --arm_checkpoint 051_bc_arm_gelu_e70_h512_l3_3min_working.pt \
  --grasp_checkpoint robotnik_sensor_bc_v2/grasp_sensor_residual_bc_v4.npz \
  --local_drop --target 2.47 3.03 1.00 --dropoff_xy 2.47 3.03 \
  --drop_zone_radius 0.32 --drop_zone_release_margin 0.05 \
  --drop_zone_approach_margin 0.04 --carry_timeout 180 --max_steps 240000 \
  --placement_servo_force 35 --placement_servo_position_gain 0.32 \
  --placement_servo_speed 0.20 --placement_joint_step 0.035 \
  --place_on_ground --ground_ball_center_z 0.50 --ground_place_speed_max 0.18 "$@" \
  2>&1 | tee v20_release_gate_diagnostic.log
