ROBOTNIK: LEARNED GRASP TARGETS + LEARNED REVERSE WHEEL COMMANDS
==================================================================

GOAL: approach ball with existing base/arm ML models, close fingers using a
newly-trained BC network, reverse 25 cm using that network. NO table/box/forward.

HONEST LIMITATION: this trains the 6 commanded finger targets and the 4 reverse
wheel commands from V10 demonstrations. V10's phase machine, safety guards,
arm-pose stabilizer, and virtual PyBullet fixed ball attachment remain. It is
NOT an end-to-end or a true finger-contact grasp policy. Producing a true
physical grasp requires valid contact-grasp demonstrations and new training.

DO NOT delete or modify run_grasp_carry_v10_centered.py. The generator leaves
it untouched and refuses to overwrite generated files.

SETUP
-----

conda activate comp0245
cd ~/Desktop/UCL/ML/ML-RAI-affan-latest/final/robotnik_arm_task
unzip ~/Downloads/robotnik_grasp_reverse_bc_kit.zip -d .
export PYTHONPATH="$HOME/Desktop/UCL/ML/RoboEnv/simulation_and_control"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python make_ml_grasp_reverse.py
python -m py_compile run_grasp_reverse_ml_record.py run_grasp_reverse_ml_eval.py

RECORD TRAINING DEMONSTRATIONS
------------------------------
Use known-working V10 with the same teacher actions. The new script merely
records the six finger targets and four wheel commands. Collect several
successful complete runs. Do not train on runs that did not reach FINISHED.

CUDA_VISIBLE_DEVICES=-1 python -u run_grasp_reverse_ml_record.py \
  --base_checkpoint 060_bc_base_gelu_e70_h512_l3_16min_working.pt \
  --arm_checkpoint 051_bc_arm_gelu_e70_h512_l3_3min_working.pt \
  --target 2 2 1.0 --max_steps 25000 --record_npz bc_demo_01.npz \
  2>&1 | tee bc_demo_01.log

You can record variations around the working goal if they are successful:

CUDA_VISIBLE_DEVICES=-1 python -u run_grasp_reverse_ml_record.py \
  --base_checkpoint 060_bc_base_gelu_e70_h512_l3_16min_working.pt \
  --arm_checkpoint 051_bc_arm_gelu_e70_h512_l3_3min_working.pt \
  --target 1.98 2.0 1.0 --max_steps 25000 --record_npz bc_demo_02.npz \
  2>&1 | tee bc_demo_02.log

CUDA_VISIBLE_DEVICES=-1 python -u run_grasp_reverse_ml_record.py \
  --base_checkpoint 060_bc_base_gelu_e70_h512_l3_16min_working.pt \
  --arm_checkpoint 051_bc_arm_gelu_e70_h512_l3_3min_working.pt \
  --target 2.02 2.0 1.0 --max_steps 25000 --record_npz bc_demo_03.npz \
  2>&1 | tee bc_demo_03.log

Inspect every run: grep -E "FINISHED|BC DATA SAVED" bc_demo_*.log
If some variants fail, collect more demonstrations at goals that work.

TRAIN A NEW NETWORK (does not retrain the original two)
-------------------------------------------------------
python grasp_reverse_bc.py --data bc_demo_01.npz bc_demo_02.npz bc_demo_03.npz \
  --out grasp_reverse_bc.pt --epochs 100

INFER WITH THREE MODELS
------------------------
CUDA_VISIBLE_DEVICES=-1 python -u run_grasp_reverse_ml_eval.py \
  --base_checkpoint 060_bc_base_gelu_e70_h512_l3_16min_working.pt \
  --arm_checkpoint 051_bc_arm_gelu_e70_h512_l3_3min_working.pt \
  --grasp_checkpoint grasp_reverse_bc.pt \
  --target 2 2 1.0 --max_steps 25000 \
  2>&1 | tee ml_grasp_reverse_test.log

Check: grep -E "LEARNED GRIP|ML CLOSE|ML REVERSE|BALL SECURED|FINGERS CLOSED|FINISHED" ml_grasp_reverse_test.log

VALIDATE:
- Observe gripper and ball in PyBullet; text alone cannot prove contact.
- Verify close and reverse actually happen.
- Compare V10, which is the unchanged recovery baseline.
- Frame-shuffled validation is not a test of unseen environments.
- The old V10 ball attachment is *virtual*; do not claim force grasp.

If generated scripts fail due to source anchors, send output and the relevant
source excerpt; the generator intentionally refuses speculative edits.
