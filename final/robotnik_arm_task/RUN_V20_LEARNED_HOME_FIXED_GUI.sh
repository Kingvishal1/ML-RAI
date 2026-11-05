#!/usr/bin/env bash
# V20 repaired learned home transition + pretrained arm-policy return to raised home EE pose.
# Original V20 controller file is NOT overwritten.
set -euo pipefail
[[ -d robotnik_sensor_bc_v2 ]] || { echo "Run this from the original V20 robotnik_arm_task root" >&2; exit 2; }
export PYTHONPATH="$PWD:$HOME/Desktop/UCL/ML/RoboEnv/simulation_and_control${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u robotnik_sensor_bc_v2/robotnik_expert_controller_learned_return.py \
 --carry_controller expert \
 --base_checkpoint 060_bc_base_gelu_e70_h512_l3_16min_working.pt \
 --arm_checkpoint 051_bc_arm_gelu_e70_h512_l3_3min_working.pt \
 --grasp_checkpoint robotnik_sensor_bc_v2/grasp_sensor_residual_bc_v4.npz \
 --target 2.5 3.0 1.0 \
 --dropoff_xy 3.0 3.5 --drop_zone_radius 0.50 \
 --carry_timeout 220 --max_steps 350000 --place_on_ground \
 --ground_ball_center_z 0.55 --ground_place_speed_max 0.095 \
 --return_home_controller learned --learned_home_ee_tolerance 0.06 \
 --keep_simulation_open "$@"
