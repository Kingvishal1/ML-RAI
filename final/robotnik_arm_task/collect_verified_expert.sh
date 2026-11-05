#!/usr/bin/env bash
# Run inside robotnik_arm_task root with conda env comp0245 activated.
set -euo pipefail
mkdir -p expert_demos
ID="$(date +%Y%m%d_%H%M%S)_${BASHPID}"
NPZ="expert_demos/expert_${ID}.npz"
LOG="expert_demos/expert_${ID}.log"
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
  --record_expert_npz "$NPZ" \
  "$@" 2>&1 | tee "$LOG"
if [[ ! -e "$NPZ" ]]; then
  echo 'No NPZ trace was saved. Review log for early crash.' >&2
  exit 2
fi
python - "$NPZ" <<'PY'
import sys
import numpy as np
path=sys.argv[1]
with np.load(path,allow_pickle=False) as z:
    good=all(bool(z[key].item()) for key in ('physical_success_verified','ball_released_verified','no_virtual_attachment'))
    print('Expert data:',path)
    print('Verified pick/carry/drop:',good)
    print('Carry expert frames:',len(z['obs']),'release expert frames:',len(z['release_obs_17']))
    if not good:
        print('DO NOT TRAIN ON THIS FAILED DEMONSTRATION.')
        sys.exit(3)
PY
