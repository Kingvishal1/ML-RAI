# 🤖 Behavioral Cloning for Multi-Object Mobile Pick and Drop Robot


![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![PyBullet](https://img.shields.io/badge/PyBullet-RoboEnv-1E88E5)
![Pinocchio](https://img.shields.io/badge/Pinocchio-kinematics-6A1B9A)
![Robot](https://img.shields.io/badge/Robot-Robotnik%20Summit%20XL-F57C00)
![Optuna](https://img.shields.io/badge/Optuna-tuning-2C3E8F)

Behavioral Cloning (BC) investigates how a mobile manipulator can learn navigation and arm-control behaviour from expert demonstrations. Built around the **Robotnik Summit XL** mobile base, a **7-DoF robotic arm**, and a **three-finger gripper**, the project combines data-driven robot control with simulation-based testing and multi-target task planning.

Instead of manually specifying every navigation and reaching action, the project records expert state–action trajectories and uses **behaviour cloning** to train separate neural policies for the **mobile base** and **robot arm**. A further learned **target-selection policy** chooses which object to attempt next in a scene containing multiple balls and assigned drop zones. These components execute in a closed-loop RoboEnv/PyBullet simulation, with Pinocchio providing robot kinematics.

## ⚡ How It Works (TL;DR)

**Expert Robotnik control** generates state/action demonstrations → **behaviour cloning** trains base and arm policies → a separate **neural selector** learns the next-target ordering → **RoboEnv/PyBullet** runs closed-loop reaching and multi-ball tasks.

## ⚙️ Tech Stack

**Robotics and simulation:** Robotnik Summit XL · PyBullet · RoboEnv (`simulation_and_control`) · Pinocchio  
**Machine learning:** PyTorch · behaviour cloning · multilayer perceptrons (MLPs)  
**Data and experiments:** NumPy · Matplotlib · JSON scene configurations

## 🧩 Design & Implementation

### 1. 🦾 Robot and Physics Simulation

`robotnik_arm_task.py` creates the RoboEnv simulation from `configs/robotnik_arm.json`. The Robotnik platform has a **four-wheel base, a seven-joint arm and six actuated finger joints** (two joints for each of three fingers). `setup_kinematics()` constructs the Pinocchio representation used for arm/end-effector calculations. Robot assets are under `models/`, while robot/simulator settings are under `configs/`.

### 2. 🎮 Expert Control and Demonstration Generation

`robotnik_arm_control.py` contains reference movement/control logic, and `robotnik_arm_task.py` defines the reference task. The expert produces trajectories with measured robot state and corresponding actuator commands. `training/collect_data.py` records demonstrations, including optional disturbance/noise settings for robustness studies.

The full expert action layout comprises **four wheel commands** and **seven arm torques**. The gripper is controlled separately from the base/arm behaviour-cloning outputs.

### 3. 🧠 Behaviour-Cloning Models

Demonstrations are processed into base-navigation and arm-reaching examples by `training/split_dataset.py`. MLP regressors are implemented through `regressors.py` and `activations.py`; `training/train_regressor.py` fits the policies. Checkpoints store learned parameters and the feature/normalisation information needed by `run_policy.py` during inference.

| Policy | Responsibility | Checkpoint |
|---|---|---|
| **Base BC** | Navigation commands from robot/target observations | `final_navigation_policy.pt` |
| **Arm BC** | Reaching control of the seven-joint arm | `final_manipulation_policy.pt` |
| **Target selector** | Next-ball choice from a multi-target scene | `final_selector_policy.pt` |

### 4. 🎯 Multi-Target Selection and Pick-and-Place

`eval/run_policy_sim.py` implements the main rollout. In multi-target mode it reads `balls`, `zones`, and `assignments` from JSON (or generates a random scene), spawns physical ball objects and zone markers, and repeatedly:

1. Forms the current scene observation and chooses a remaining ball via the trained selector (or expert shortest-path ordering when selected).
2. Uses the base/arm controller to approach the selected ball and execute the task-specific pick-and-place routine.
3. Attempts to return the arm towards its home pose before proceeding to another target.
4. Records the visit order, grasp/delivery outcomes and task statistics.

**Methodological note:** the selector decides **which target to visit**. It does not directly predict individual finger actions. Manipulation and phase transitions include explicit programmed control.

### 5. 📊 Evaluation and Research Experiments

`eval/evaluate.py` runs multiple targets/episodes; `eval/parse_eval_log.py` and `eval/utils.py` support result analysis. `experiments/` covers activation sweeps, ablations, model comparisons and dataset scaling. `metrics/` contains feature-importance and threshold-sweep utilities.

## 📁 Repository Guide

```text
ML-RAI/
├── README.md                          # Main project documentation
└── final/
    └── robotnik_arm_task/
        ├── robotnik_arm_task.py        # Simulator setup and reference task
        ├── robotnik_arm_control.py     # Expert base/arm control
        ├── run_policy.py              # Policy loading and predictions
        ├── regressors.py              # Neural regression architectures
        ├── activations.py             # Activation functions
        ├── requirements.txt           # Minimal dependency listing
        ├── *.pt                       # Pretrained checkpoints
        ├── configs/                   # Robot configuration files
        ├── models/                    # Robot URDFs and geometry
        ├── eval/
        │   ├── run_policy_sim.py       # Main demonstration entry point
        │   ├── evaluate.py            # Repeated quantitative evaluation
        │   ├── example_config.json    # Fixed multi-ball scene
        │   ├── parse_eval_log.py       # Evaluation-log processing
        │   └── utils.py               # Evaluation helpers
        ├── training/
        │   ├── collect_data.py         # Expert demonstration collection
        │   ├── split_dataset.py        # Base/arm dataset preparation
        │   ├── train_regressor.py      # Behaviour-cloning training
        │   ├── dagger_collect.py       # Expert-labelled policy-state data
        │   ├── merge_datasets.py       # Dataset merging
        │   ├── collect_selector_data.py# Selector training data
        │   └── train_selector.py       # Target-selection model training
        ├── experiments/               # Ablation, activation and scaling studies
        └── metrics/                   # Feature and threshold analyses
```


## 🚀 Getting Started

The **multi-target pick-and-place demonstration** is the primary project entry point. The repository contains the required pretrained checkpoints, so retraining is not required for this demonstration.

### 1. Prerequisites

- **Git**
- **Pixi** (Python environment manager, recommended by the COMP0245 course)
- Python dependencies managed through RoboEnv’s Pixi `ml` environment

Install Git and Pixi if you do not have them:

```bash
# Windows (PowerShell)
winget install --id Git.Git -e --source winget
irm https://pixi.sh/install-pixi.ps1 | iex
```

```bash
# macOS / Linux
curl -fsSL https://pixi.sh/install.sh | sh
```

> Install the RoboEnv `ml` environment, not just its default environment: the latter may not include PyTorch. The project commands below use the RoboEnv Pixi environment directly.

### 2. Clone RoboEnv and this repository

Clone both side by side in the same workspace folder. **RoboEnv** is the course simulator that provides `simulation_and_control` (`pb`, `MotorCommands`, `PinWrapper`).

```bash
git clone --recurse-submodules https://github.com/VModugno/RoboEnv.git
git clone --branch refactor-clean-code --single-branch https://github.com/Kingvishal1/ML-RAI.git
```

Resulting layout:

```text
workspace/
├── RoboEnv/
└── ML-RAI/
    └── final/robotnik_arm_task/
```

### 3. Create the environment

This project needs PyTorch, which RoboEnv provides in its **`ml`** environment:

```bash
cd RoboEnv
pixi install -e ml
pixi run -e ml python -c "import torch, pybullet, pinocchio, simulation_and_control; print('Dependencies OK')"
```

### 4. Run commands in the project environment

From the Robotnik project directory, use `pixi run -m` to run commands in RoboEnv's `ml` environment without activating a separate shell:

```bash
cd "$HOME/robotnik-workspace/ML-RAI/final/robotnik_arm_task"
```

### 5. ▶️ Single-Target Reaching

```bash
pixi run -m "$HOME/robotnik-workspace/RoboEnv/pixi.toml" -e ml \
  python eval/run_policy_sim.py \
  --base_checkpoint final_navigation_policy.pt \
  --arm_checkpoint final_manipulation_policy.pt \
  --target 2 2 1.0
```

**Evaluate single-target reaching:**

```bash
pixi run -m "$HOME/robotnik-workspace/RoboEnv/pixi.toml" -e ml \
  python eval/evaluate.py --headless --num_targets 20 --seed 42 \
  --base_checkpoint final_navigation_policy.pt \
  --arm_checkpoint final_manipulation_policy.pt
```

### 6. ▶️ Multi-Target Pick-and-Place

```bash
pixi run -m "$HOME/robotnik-workspace/RoboEnv/pixi.toml" -e ml \
  python eval/run_policy_sim.py --pick_and_place --config eval/example_config.json \
  --base_checkpoint final_navigation_policy.pt \
  --arm_checkpoint final_manipulation_policy.pt \
  --selector_checkpoint final_selector_policy.pt
```

**Evaluate multi-target pick-and-place:**

```bash
pixi run -m "$HOME/robotnik-workspace/RoboEnv/pixi.toml" -e ml \
  python eval/evaluate.py --pick_and_place --headless \
  --num_episodes 10 --num_targets_per_episode 3 --multi_target_seed 42 \
  --base_checkpoint final_navigation_policy.pt \
  --arm_checkpoint final_manipulation_policy.pt \
  --selector_checkpoint final_selector_policy.pt
```

The example scene configures **three balls, three drop zones and one assigned destination per ball**. During a multi-target episode, the neural selector chooses which ball to attempt, and the simulator uses the pretrained base and arm policies with the project's task-level pick-and-place logic. Add `--headless` to run without the PyBullet GUI.

**Example configuration:** `eval/example_config.json`

```json
{
  "balls": [[2.0, 1.5, 0.5], [-1.0, 3.0, 0.4], [3.5, -2.0, 0.6]],
  "zones": [[0.0, 4.0, 0.5], [-3.0, -1.0, 0.5], [4.0, 2.0, 0.5]],
  "assignments": [0, 1, 2]
}
```

## 🏋️ Training Your Own Policies

Run from `ML-RAI/final/robotnik_arm_task` using the RoboEnv `ml` environment. For the training commands below, enter `pixi shell -e ml` from `RoboEnv` first, then return to the Robotnik project directory. Tuned hyperparameters are used by default.

**1. Collect expert demonstrations (base + arm)**
```bash
python training/collect_data.py --mode random --num_targets 1000 --xy_range 5 --vary_init --noise_std 1.0 --workers 8 --seed 42
```

**2. Split into base and arm datasets**
```bash
python training/split_dataset.py --data data/<dataset_file_name>.npz --out_dir data/split --arm_full_torque
```

**3. Train the base policy**
```bash
python training/train_regressor.py --data data/split/dataset_base.npz
```

**4. Train the arm policy**
```bash
python training/train_regressor.py --data data/split/dataset_arm.npz
```

**5. Collect selector data**
```bash
python training/collect_selector_data.py --num_episodes 10000 --max_targets 3 --seed 42
```

**6. Train the selector**
```bash
python training/train_selector.py --data data/<selector_dataset_file_name>.npz
```

**7. Run your trained policies**
```bash
python eval/run_policy_sim.py --pick_and_place --config eval/example_config.json --base_checkpoint <base>.pt --arm_checkpoint <arm>.pt --selector_checkpoint <selector>.pt
```