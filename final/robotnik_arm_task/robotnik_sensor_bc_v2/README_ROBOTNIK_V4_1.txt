ROBOTNIK V4.1 — SINGLE-BALL ML GRAVITY-HOLD TEST
================================================

What was changed relative to V4
--------------------------------
1. Removed the add_target_visual() call. Previously this made a separate static
   target marker at the same location as the physical red ball. When the ball
   fell/slipped, the marker stayed suspended and looked like another ball.
   The new runner creates exactly one VISIBLE red ball with a collision shape.
   The separate gravity-check object has NO visual shape and is removed early.
2. Added configurable PHYSICAL lateral friction for the red ball and the six
   finger links (defaults 1.5 each) and 80 contact solver iterations.
   These are physics settings, NOT scripted gripper targets.
3. Requires 0.25 s of continuous 2-finger-group contact before dynamic release
   (previously 0.12 s); checks gravity hold with the same strict criteria.
4. Logs contact normal force, first contact loss, first 5 mm slip, ball speed,
   release timing, positional drift, and gravity hold outcome.

What was NOT changed
--------------------
- Existing base and arm .pt policies.
- Existing V4 finger network and weights: grasp_sensor_residual_bc_v4.npz.
- sensor_bc_v4.py inference logic: all six motor targets still come from the
  learned neural network (measured-q plus its learned delta).
- No manual finger sweep, forced squeeze, virtual attachment, weld, magnet,
  artificial upward force, or fixed base constraint.
- Ball is held static during approach/contact calibration and becomes dynamic
  only after sustained contact. This is still a HYBRID experiment, not
  end-to-end ML grasping. A stable grasp has not been demonstrated in the
  uploaded log; the final outcome requires execution on your computer.

Install ZIP (use Ubuntu, NOT a different device)
----------------------------------------------
cd ~/Desktop/UCL/ML/ML-RAI-affan-latest/final/robotnik_arm_task
conda activate comp0245
unzip -o ~/Downloads/robotnik_ml_grasp_v4_1.zip -d robotnik_sensor_bc_v2/
ls -lh robotnik_sensor_bc_v2/run_grasp_ml_v4_1.py
ls -lh robotnik_sensor_bc_v2/sensor_bc_v4.py
ls -lh robotnik_sensor_bc_v2/grasp_sensor_residual_bc_v4.npz

You can install JUST the .py file if your original V4 model and sensor_bc_v4.py
are already present:
cp ~/Downloads/run_grasp_ml_v4_1.py robotnik_sensor_bc_v2/

Run from robotnik_arm_task
-------------------------
PYTHONPATH="$PWD:$HOME/Desktop/UCL/ML/RoboEnv/simulation_and_control${PYTHONPATH:+:$PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u robotnik_sensor_bc_v2/run_grasp_ml_v4_1.py \
  --base_checkpoint 060_bc_base_gelu_e70_h512_l3_16min_working.pt \
  --arm_checkpoint 051_bc_arm_gelu_e70_h512_l3_3min_working.pt \
  --grasp_checkpoint robotnik_sensor_bc_v2/grasp_sensor_residual_bc_v4.npz \
  --target 2 2 1.0 --max_steps 30000 \
  --record_ml_npz "grasp_v4_1_predictions_$(date +%Y%m%d_%H%M%S).npz" \
  2>&1 | tee grasp_ml_v4_1.log

What to look for in log
-----------------------
V4.1 VISUAL: one physical ball
V4_1_RELEASE_TO_GRAVITY
V4_1_FIRST_CONTACT_LOSS (if any)
V4_1_FIRST_5MM_SLIP (if any)
V4_1_GRAVITY_HOLD
V4_1_RESULT: GRAVITY HOLD PASS or NOT VERIFIED / FAIL

If it still slips, send grasp_ml_v4_1.log. Contact loss can be a real failure:
only two fingertips were observed to contact in the V2 diagnostic, and the
existing V4 teacher data is a virtual-attachment trajectory. Changing physical
friction cannot guarantee a stable learned three-finger hold. Do not label a
failed V4.1 run a successful grasp or automatically train on its predictions
as successful expert labels.

Physics realism note
--------------------
Friction 1.5 is a test material assumption, not an experimentally measured
coefficient for your robot. For sensitivity you may rerun with
--ball_friction 1.2 --finger_friction 1.2
without changing ML weights. Real hardware requires validation of contact
friction, finger effort, and object mass.
