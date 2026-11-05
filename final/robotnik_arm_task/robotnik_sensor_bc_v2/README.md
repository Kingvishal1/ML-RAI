# Robotnik sensor-only BC v2 — trained close/reverse subpolicy

This package contains a real trained 17→64→64→10 tanh neural network, exported
as NumPy weights. It is **not a trained end-to-end navigation/grasp/carry model**.
It learns six finger position targets and four wheel velocity commands from the
uploaded demonstration. NumPy inference does not require scikit-learn or PyTorch;
your existing base/arm runner still requires its normal PyTorch/RoboEnv stack.

## What you received

- `grasp_reverse_sensor_bc_v2.npz`: trained weights, normalization, action limits.
- `sensor_bc.py`: portable inference and reproducible training code.
- `run_grasp_reverse_contact_v2.py`: separate physical-contact diagnostic, based
  on the uploaded runner; no original files or base/arm checkpoints are replaced.
- `grasp_reverse_sensor_bc_v2.metrics.json`: offline validation results/limitations.
- `verify_sensor_bc.py`: checks the export and selected source-level properties.
- `AUDIT.md`: specific findings in the uploaded code and what remains.

## Install and run on your laptop

Extract the ZIP into a separate subfolder inside your existing task directory:

```bash
conda activate comp0245
cd ~/Desktop/UCL/ML/ML-RAI-affan-latest/final/robotnik_arm_task
unzip -n ~/Downloads/robotnik_sensor_bc_v2.zip
```

Verify against your existing demonstration:

```bash
python robotnik_sensor_bc_v2/verify_sensor_bc.py --data bc_demo_01.npz
```

Run the physical-contact diagnostic. `PYTHONPATH` makes the existing project
modules accessible from the subfolder. Its existing `robotnik_arm_task.py`
continues locating your JSON configuration and assets in the project directory.
If `physical_trace_01.npz` already exists, use a new filename.

```bash
PYTHONPATH="$PWD:$HOME/Desktop/UCL/ML/RoboEnv/simulation_and_control${PYTHONPATH:+:$PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u robotnik_sensor_bc_v2/run_grasp_reverse_contact_v2.py \
  --base_checkpoint 060_bc_base_gelu_e70_h512_l3_16min_working.pt \
  --arm_checkpoint 051_bc_arm_gelu_e70_h512_l3_3min_working.pt \
  --grasp_checkpoint robotnik_sensor_bc_v2/grasp_reverse_sensor_bc_v2.npz \
  --target 2 2 1.0 --max_steps 25000 \
  --record_full_npz physical_trace_01.npz \
  2>&1 | tee physical_contact_v2.log
```

Send `physical_contact_v2.log` and `physical_trace_01.npz` back for the next step.
The runner needs the existing `robotnik_arm_control.py`, `run_policy.py`,
`regressors.py`, `activations.py`, config, URDF/assets and checkpoints. Those were
not in the uploaded archive and are intentionally not replaced or invented.
The finger actuation helper is inherited unchanged; its callback/actuation
behaviour must be checked in the simulator because its source was not uploaded.

## What changed

The close/reverse network receives only ball-to-EE displacement in the base yaw
frame, six finger angles, six finger velocities and two base-frame velocities.
It has no scripted phase input, clock, distance-since-handover or attachment flag.
After the existing approach handover it predicts fingers AND wheels each step;
there is no fixed 0.8-second close timer in this runner.

Physical scale floors prevent nearly constant training features from being
normalized by tiny standard deviations. Predicted actions are clipped to the
training action ranges. That clamp is deterministic and is not learned safety
or evidence of robustness. In particular, it prevents large positive wheel
commands outside the recorded range; this does not prove reversal was learned
robustly. Raw, pre-clamp errors are reported separately.

The runner makes the ball dynamic on a table; removes ball attachment, the base
world lock, and ball/robot collision disabling. Existing phase-based navigation,
approach alignment, wheel holding and arm PD holding remain: the whole runner
is still hybrid. Removing the base lock changes the task and may expose drift.
A signed 25cm backward-distance cutoff is an evaluation limit, NOT a learned
stop or success signal. Time-limit termination likewise does not prove success.

## Training evidence

Source: one 773-frame synthetic/virtual-attachment teacher episode. Train: 559
frames. Holdout: 194 chronological frames, chosen as the last quarter within
each phase, with ten frames removed between each train/holdout segment.
Twenty frames are excluded. This checks within-trajectory imitation only.
Phase labels are used only for this offline split, never as network inputs.

Raw mean absolute errors, averaged across the relevant output joints:
- Finger targets: approximately 0.0037 radians.
- Wheel velocities: approximately 0.0596 rad/s.

The holdout was not used to select the exported weights. The model was fitted
on the 559 training frames using L-BFGS with fixed seed 42; 305 iterations.
About 47.9% of individual holdout output components are clipped by the support
bounds, often at endpoint finger targets. Consult the raw metrics rather than
using bounded metrics to imply improvement in physical control.

No physical or closed-loop validation was possible here: this runtime has no
PyBullet, Pinocchio, RoboEnv, Robotnik assets or base/arm checkpoints.

## Retrain (optional, not needed to run the included model)

Requires NumPy, SciPy and scikit-learn in the training environment. Use a new
output name; the trainer refuses to overwrite existing weights.

```bash
python robotnik_sensor_bc_v2/sensor_bc.py \
  --data bc_demo_01.npz --out grasp_reverse_sensor_retrained.npz
```

When multiple legacy-format episodes are supplied, the final supplied episode
is held out entirely. Provide independently collected initial conditions; simply
copying one file does not create independent evidence. This trainer handles
legacy 24-input/10-action close/reverse data only, not the new full-task traces.

## Full physical trace schema

Optional `physical_trace_01.npz` records every five physics steps:
- `obs`: 63 inputs: base xyz(3), yaw cos/sin(2), base linear velocity(3), base
  angular velocity(3), motor q(17), motor qd(17), EE xyz(3), ball xyz(3), ball
  linear velocity(3), requested goal xyz(3), finger contact bits(6).
- `action`: wheel velocities(4), arm torques(7), finger position targets(6).
- `time`: timestamps for trajectory analysis, not policy inputs.
- `virtual_attachment`: false; `success_verified`: false.

Finger contact bits follow the existing URDF-discovered finger_joint_ids order.
Motor q/qd ordering is inherited from the simulator; verify the six finger
entries align with that order before training from new traces. Actions are the
commands/targets requested, not measured motor efforts. Ordinary interpreter
exit saves the trace; forced termination may not. A trace can be an unsuccessful
attempt. Do not relabel it a successful demonstration without reviewing contact,
lift, object transport, slippage and final stability over time.

## Remaining work for the user's full objective

First establish successful physical pickup/carry with enabled collision geometry
and actual finger actuation. Then record varied complete episodes, including
navigation, reach, grasp, lift, reverse, final holding/stopping and recovery.
The current uploaded data contains neither arm-torque labels nor approach/stop
examples, so those behaviours cannot be supervised from it. Train a full or
hierarchical policy with learned switching from those episodes, reserve complete
unseen episodes, and measure physical task success without virtual attachment.
The existing base/arm models can be retained as warm starts/teachers; do not
throw away the working baseline or claim the present diagnostic is that final model.
