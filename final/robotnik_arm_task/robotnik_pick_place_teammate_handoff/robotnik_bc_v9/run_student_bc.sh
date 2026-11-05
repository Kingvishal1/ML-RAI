#!/usr/bin/env bash
# Closed-loop student BC evaluation. Never silently uses expert carry/release.
set -euo pipefail
[[ -d robotnik_sensor_bc_v2 && -d robotnik_submission_tools ]] || {
  echo 'Run from robotnik_arm_task root after install.sh' >&2; exit 2;
}
CANDIDATES="${1:?Usage: bash robotnik_submission_tools/run_student_bc.sh TRAINED_DIR [--target ...]}"
shift
CARRY="$CANDIDATES/carry_candidate_bc.npz"
RELEASE="$CANDIDATES/release_candidate_bc.npz"
[[ -f "$CARRY" && -f "$RELEASE" ]] || { echo 'Train candidate weights first' >&2; exit 2; }
export PYTHONPATH="$PWD:$HOME/Desktop/UCL/ML/RoboEnv/simulation_and_control${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u robotnik_sensor_bc_v2/robotnik_bc_controller.py \
  --carry_controller bc --release_controller bc \
  --carry_bc_checkpoint "$CARRY" --release_bc_checkpoint "$RELEASE" \
  --base_checkpoint 060_bc_base_gelu_e70_h512_l3_16min_working.pt \
  --arm_checkpoint 051_bc_arm_gelu_e70_h512_l3_3min_working.pt \
  --grasp_checkpoint robotnik_sensor_bc_v2/grasp_sensor_residual_bc_v4.npz \
  --target 2.5 3.0 1.0 --dropoff_xy 3.0 3.5 \
  --carry_timeout 220 --max_steps 350000 "$@"
