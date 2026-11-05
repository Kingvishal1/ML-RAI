# Grasp-only air-ball attempt

This adds `run_grasp_only_air_v1.py` without changing the earlier diagnostic.
It attempts to reach and close around the ball, then stops at a sustained
multi-finger contact candidate. Reverse motion is disabled.

The base/arm approach and the finger closer are neural-network inferences, but
the runner remains hybrid: the base-to-arm switch and final grasp handover are
scripted, and the arm uses a joint-hold correction during closure. The closer
was trained from a single teacher trajectory with virtual ball attachment; it
has not been trained on verified physical grasps. Treat this as a diagnostic
attempt, not a proven grasp model. The candidate flag requires at least two
finger groups to contact the ball, close ball-to-EE distance, and low relative
motion for 0.75 seconds. Verify the ball stays between the fingers in the GUI.

The board/support and virtual attachment are removed. The ball is dynamic and
its weight is cancelled with an upward force, so this tests contact on a
suspended, effectively weightless ball rather than a normal-gravity pickup.

Run from the project root after extracting this archive:

```bash
conda activate comp0245
cd ~/Desktop/UCL/ML/ML-RAI-affan-latest/final/robotnik_arm_task
PYTHONPATH="$PWD:$HOME/Desktop/UCL/ML/RoboEnv/simulation_and_control${PYTHONPATH:+:$PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u robotnik_sensor_bc_v2/run_grasp_only_air_v1.py \
  --base_checkpoint 060_bc_base_gelu_e70_h512_l3_16min_working.pt \
  --arm_checkpoint 051_bc_arm_gelu_e70_h512_l3_3min_working.pt \
  --grasp_checkpoint robotnik_sensor_bc_v2/grasp_reverse_sensor_bc_v2.npz \
  --target 2 2 1.0 --max_steps 25000 \
  2>&1 | tee grasp_only_air_v1.log
```

Leave out `--headless` to see the simulator. Evidence of a candidate is still
not force-closure or robustness proof; inspect contacts and the ball's motion.
If the fingers do not move, check the finger target angles and the simulator's
`configure_gripper_fingers` position-control callback.
