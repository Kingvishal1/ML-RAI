#!/usr/bin/env bash
# Run this from robotnik_arm_task after install_into_project.sh.
set -euo pipefail
[[ -d robotnik_sensor_bc_v2 ]] || { echo 'Run this from robotnik_arm_task project root' >&2; exit 2; }
export PYTHONPATH="$PWD:$HOME/Desktop/UCL/ML/RoboEnv/simulation_and_control${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u robotnik_sensor_bc_v2/robotnik_expert_controller.py \
  --carry_controller expert \
  --base_checkpoint 060_bc_base_gelu_e70_h512_l3_16min_working.pt \
  --arm_checkpoint 051_bc_arm_gelu_e70_h512_l3_3min_working.pt \
  --grasp_checkpoint robotnik_sensor_bc_v2/grasp_sensor_residual_bc_v4.npz \
  --target 2.5 3.0 1.0 \
  --dropoff_xy 3 3.5 \
  --carry_timeout 220 \
  --max_steps 350000 \
  "$@"
