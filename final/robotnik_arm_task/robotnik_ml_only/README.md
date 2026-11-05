# Robotnik: full-action learned-policy training

**This is an unvalidated training implementation, not a trained successful grasp
model. No full-task checkpoint is included.** The actor must be trained on the
user's Robotnik simulator. A smoke test is required before a long run.

## What is different

One neural actor maps 66 sensor/goal values to 17 continuous actions: four wheel
velocity targets, seven arm torques and six finger position targets. All outputs
are predicted on every policy step. There is no rotate/drive phase input,
base/arm handover, proximity-triggered gripper activation, close timer, programmed
reverse speed, arm-hold PD correction or ball attachment in the new controller.

The simulator still implements wheel velocity servos and finger position servos.
These are actuator interfaces, not learned low-level torque controllers. If the
coursework requires learned torque control for every actuator, this interface
would need to be changed. Task rewards, action bounds, episode reset, termination
and safety limits are deterministic environment rules, not learned motor choices.

Your previous files and checkpoints are unchanged. Their successful approach
behaviour is used via the recorded trace for imitation warm-start, rather than
loading incompatible split-policy weights into the new actor. The trace's stalled
section after 13.3s is excluded. It contains no learned or successful grasp.

## Scene: ball in the air

There is no wooden board. The 50g ball is dynamic from the beginning and receives
a constant upward force of mass × 9.81 each physics step. This cancels its weight
assuming the standard simulator gravity of -9.81m/s². It is never attached to the
robot, frozen during approach or released by a phase rule. Collisions remain
active. This is a **weightless-object benchmark**, not ordinary-gravity pickup.
Even successful transport here would not establish normal-gravity grasping.

The goal is to move the ball 25cm horizontally toward the robot's initial base
position, with finger contact and stable holding. No motor command explicitly
instructs a reverse. The reward supplies approach/contact/transport objectives.
Candidate success additionally requires at least 20cm base retreat toward the
initial base position, measured from the first qualifying finger-contact event.
The reward encourages retreat while maintaining contact. This scoring reference
never changes the motor commands or activates a different policy. It does not
force the exact earlier scripted path, and task success is not guaranteed.

## Install

From the existing project directory:

```bash
conda activate comp0245
cd ~/Desktop/UCL/ML/ML-RAI-affan-latest/final/robotnik_arm_task
unzip -n ~/Downloads/robotnik_ml_only_training.zip
python robotnik_ml_only/check_offline.py
```

This uses the existing PyTorch, NumPy, RoboEnv, PyBullet and Pinocchio environment.
No Stable-Baselines or Gym installation is needed. The task's original
`robotnik_arm_task.py`, `robotnik_arm_control.py`, configuration and URDF/assets
must remain available in the parent project. The supplied JSON refers to
`summit_xl_arm_description/summit_xl_complete.urdf` and its Pinocchio counterpart;
those assets were not uploaded and cannot be executed in this workspace.

## First run: short training smoke test

```bash
PYTHONPATH="$PWD:$HOME/Desktop/UCL/ML/RoboEnv/simulation_and_control${PYTHONPATH:+:$PYTHONPATH}" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u robotnik_ml_only/train_full_policy.py \
  --warmstart_trace air_trace_01.npz \
  --out ml_only_smoke_01 --bc_epochs 20 --steps 2048 \
  2>&1 | tee ml_only_smoke_01.log
```

The smoke test checks environment stepping and warm-start fitting. If at least
80% of completed episodes safety-stop with no successes, it halts before a PPO
update and writes the exact safety reason to the episode log. **2,048 decisions are not expected to teach
the complete task.** Send the log if this fails. Output folders must be new;
the trainer refuses to overwrite an existing folder.

A longer run, after the smoke test succeeds:

```bash
PYTHONPATH="$PWD:$HOME/Desktop/UCL/ML/RoboEnv/simulation_and_control${PYTHONPATH:+:$PYTHONPATH}" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u robotnik_ml_only/train_full_policy.py \
  --warmstart_trace air_trace_01.npz \
  --out ml_only_training_01 --bc_epochs 100 --steps 1000000 \
  2>&1 | tee ml_only_training_01.log
```

A million policy decisions correspond to 20 million physics steps at the default
20-step action repeat. Runtime and success are unknown. PPO from approach-only
initialization may fail to discover a stable grasp. Do not repeatedly spend long
runs without examining the episode log and intermediate rollouts. There is no
resume feature in this initial implementation; `latest.pt` supports evaluation,
not exact optimizer-state continuation.

## Policy test

```bash
PYTHONPATH="$PWD:$HOME/Desktop/UCL/ML/RoboEnv/simulation_and_control${PYTHONPATH:+:$PYTHONPATH}" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u robotnik_ml_only/evaluate_full_policy.py \
  --checkpoint ml_only_smoke_01/latest.pt \
  --episodes 1 --out ml_only_smoke_eval_01.json
```

For a trained model, substitute its checkpoint and use at least 10 evaluation
episodes. The default evaluation seed 1042 differs from training seed 42.
Evaluation uses deterministic actor outputs. A new evaluation output path is
required. `approach_warmstart.pt` is explicitly marked `warm_start_only`; the PPO
checkpoints are marked `ppo_unvalidated`. Neither is automatically certified.

## Task scoring and observations

The 66 inputs are the previous full-trace schema (63 values) plus a fixed 3D ball
transport goal. Wheel angles are wrapped, features have fixed physical scales,
and the actor is a 256×256 tanh MLP. A separate critic estimates value. Outputs
use tanh and map to bounded physical motor commands. Finger bounds come from the
actual URDF and the environment checks joint ordering instead of guessing it.

Approach reward uses decreasing EE-ball distance; contact reward requires at
least two distinct finger groups. Goal-progress reward is conditioned on that
contact criterion. A candidate terminal success requires goal error <4cm, two
finger groups contacting, EE-ball distance <12cm, low relative motion, low ball,
base and arm speeds, and at least 20cm base retreat, all maintained for one second. This is a task-scoring
heuristic, not proof of opposing force closure. Visual inspection and robustness
tests remain necessary, including ruling out pushing/contact exploitation.

Policy decisions happen every 20 physics steps. Safety terminates an episode on
large joint speed, base speed/travel, nonfinite state or a ball below z=0.1m. Safety
termination does not replace the actor with an expert or mark the trial successful.
The environment resets Bullet state and refreshes observations; behaviour of any
additional RoboEnv internal state across resets remains to be verified locally.

## Verification performed here

- All Python files compile.
- Pure-NumPy sensor shape/scaling and wheel-angle wrapping checks passed.
- GAE tests passed, including terminal versus time-limit bootstrapping and
  preventing returns from leaking across episodes.
- A mocked simulator verifies that all 17 actor outputs are routed directly to
  their corresponding commands/targets without phase-dependent overrides.

PyTorch training, actual RoboEnv stepping and physical success were NOT tested
here. This workspace lacks PyTorch, PyBullet, Pinocchio and the robot assets.
The uploaded `collision_enable: [0]` setting is not interpreted as global
robot-object collision disabling without inspecting the actual simulator loader.
If contact never occurs, inspect collision geometry/filtering before training more.
