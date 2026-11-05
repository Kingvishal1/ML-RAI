ROBOTNIK V4 - EXPERIMENTAL LEARNED FINGER CLOSURE
=================================================

Purpose
-------
V3's trained absolute-position finger model stalled around 0.2-0.35 rad.
V4 uses a NEW trained residual MLP: its six outputs are learned finger
position INCREMENTS conditioned on 17 sensor values. The issued position
command is the current measured joint position plus the neural prediction.
There is NO scripted closing sweep, minimum closing increment, post-contact
squeeze, hidden manual action override, virtual attachment, or reverse action.
Motor joint-limit/action-support clipping remains as a safety guard.

Truthful scope
--------------
- Base navigation is still your existing trained .pt model.
- Arm positioning is still your existing trained .pt model.
- Handover and arm holding are HYBRID, not end-to-end ML.
- The V4 finger model was trained using ONLY the supplied single
  bc_demo_01.npz recording, a VIRTUAL-ATTACHMENT teacher demonstration.
- Added stopped-state and geometry examples are SYNTHETIC perturbations
  of that source trajectory, not new successful physical grasps.
- The V4 model has passed a SIMPLIFIED OFFLINE motor-response check only.
- Holding and release must be validated in PyBullet; no claim of success.
- No physical validation or reverse transport is included in this V4 test.

Archive files
-------------
run_grasp_ml_v4.py             Full robot simulation evaluation (based on V3).
sensor_bc_v4.py               Training and exact inference for the learned
                              17-input, 6-output residual MLP.
grasp_sensor_residual_bc_v4.npz    Actual newly trained NN weights.
grasp_sensor_residual_bc_v4.metrics.json    Training provenance & audit.
bc_demo_01.npz               Original single-episode teacher training data.
verify_v4_offline.py          Physics-free smoke check. Not grasp validation.

Installation on Ubuntu
----------------------
1. Open this chat and DOWNLOAD robotnik_ml_grasp_v4.zip onto your Ubuntu PC.
2. Run these commands in a terminal:

cd ~/Desktop/UCL/ML/ML-RAI-affan-latest/final/robotnik_arm_task
conda activate comp0245
mkdir -p robotnik_sensor_bc_v2
unzip -o ~/Downloads/robotnik_ml_grasp_v4.zip -d robotnik_sensor_bc_v2/
ls -lh robotnik_sensor_bc_v2/run_grasp_ml_v4.py robotnik_sensor_bc_v2/grasp_sensor_residual_bc_v4.npz

Optional: check NN inference (not physical proof):

cd robotnik_sensor_bc_v2
python verify_v4_offline.py --checkpoint grasp_sensor_residual_bc_v4.npz
cd ..

PHYSICAL SIMULATION (from robotnik_arm_task directory)
------------------------------------------------------
PYTHONPATH="$PWD:$HOME/Desktop/UCL/ML/RoboEnv/simulation_and_control${PYTHONPATH:+:$PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u robotnik_sensor_bc_v2/run_grasp_ml_v4.py \
  --base_checkpoint 060_bc_base_gelu_e70_h512_l3_16min_working.pt \
  --arm_checkpoint 051_bc_arm_gelu_e70_h512_l3_3min_working.pt \
  --grasp_checkpoint robotnik_sensor_bc_v2/grasp_sensor_residual_bc_v4.npz \
  --target 2 2 1.0 --max_steps 30000 \
  --record_ml_npz grasp_v4_ml_predictions.npz \
  2>&1 | tee grasp_ml_v4.log

Upload grasp_ml_v4.log back to chat for diagnosis.

The physics test keeps the ball STATIC only until confirmed >=2 finger groups
contact it. It then releases the ball as a FREE 50g body under gravity with
NO weld/constraint/attachment or antigravity compensation. Passing the 2-second
gravity hold is evidence in simulation, not yet proof of robust grasping.

RETRAINING (optional / to reproduce with your original source data)
------------------------------------------------------------------
cd robotnik_sensor_bc_v2
python sensor_bc_v4.py --data bc_demo_01.npz \
   --out another_v4_checkpoint.npz --max_iter 260 --augment 8

The script refuses to overwrite an existing checkpoint. It reports that no
independent episode split and no physical success validation are available.
Re-training on the SAME single teacher demonstration does NOT solve the
missing successful physically held grasp demonstration problem.

Troubleshooting downloads
-------------------------
If ~/Downloads/robotnik_ml_grasp_v4.zip is not found, check the actual
filename and path in your browser or run:
find ~/Downloads ~/Desktop -iname '*v4*.zip' -o -iname '*v4*.py'

Do not run old run_grasp_ml_gravity_v3.py for this checkpoint. It imports
sensor_bc.py and expects the old absolute-action model format.
