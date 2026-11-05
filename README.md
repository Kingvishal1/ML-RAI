# 🤖 Behavioral Cloning for Multi-Object Mobile Pick and Drop Robot

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![PyBullet](https://img.shields.io/badge/PyBullet-RoboEnv-1E88E5)
![Pinocchio](https://img.shields.io/badge/Pinocchio-kinematics-6A1B9A)
![Robot](https://img.shields.io/badge/Robot-Robotnik%20Summit%20XL-F57C00)
![Optuna](https://img.shields.io/badge/Optuna-tuning-2C3E8F)

## Project Overview

Behavioral Cloning (BC) investigates how a mobile manipulator can learn navigation and arm-control behaviour from expert demonstrations. Built around the **Robotnik Summit XL** mobile base, a **7-DoF robotic arm**, and a **three-finger gripper**, the project combines data-driven robot control with simulation-based testing and multi-target task planning.

Instead of manually specifying every navigation and reaching action, the project records expert state–action trajectories and uses **behaviour cloning** to train separate neural policies for the **mobile base** and **robot arm**. A further learned **target-selection policy** chooses which object to attempt next in a scene containing multiple balls and assigned drop zones. These components execute in a closed-loop RoboEnv/PyBullet simulation, with Pinocchio providing robot kinematics.

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

| Policy | Responsibility | Default checkpoint |
|---|---|---|
| **Base BC** | Navigation commands from robot/target observations | `060_bc_base_gelu_e70_h512_l3_16min_working.pt` |
| **Arm BC** | Reaching control of the seven-joint arm | `051_bc_arm_gelu_e70_h512_l3_3min_working.pt` |
| **Target selector** | Next-ball choice from a multi-target scene | `084_selector_gelu_e300_h128_l3.pt` |

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
- A machine capable of running PyBullet; a graphical desktop for the GUI

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

> Pixi installs Python and every dependency (PyBullet, Pinocchio, PyTorch, `simulation_and_control`) itself — no separate Python or Conda installation is needed. Windows, Linux and macOS are supported natively; WSL is not required.

### 2. Clone RoboEnv and this repository

Clone both side by side in the same workspace folder. **RoboEnv** is the course simulator that provides `simulation_and_control` (`pb`, `MotorCommands`, `PinWrapper`).

```bash
git clone --recurse-submodules https://github.com/VModugno/RoboEnv.git
git clone --branch main --single-branch https://github.com/Kingvishal1/ML-RAI.git
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
```

### 4. Activate the environment

Do this **in every new terminal** before running the project:

```bash
cd RoboEnv
pixi shell -e ml
cd ../ML-RAI/final/robotnik_arm_task
```

### 6. ▶️ Run the main simulation — Multi-Target Pick-and-Place

From `ML-RAI/final/robotnik_arm_task`, inside the pixi shell:

```bash
python eval/run_policy_sim.py --pick_and_place --config eval/example_config.json
```

Default checkpoints: `060_bc_base_gelu_e70_h512_l3_16min_working.pt` (base), `051_bc_arm_gelu_e70_h512_l3_3min_working.pt` (arm), and `084_selector_gelu_e300_h128_l3.pt` (selector).

The example scene configures **three balls, three drop zones and one assigned destination per ball**. During a multi-target episode, the neural selector chooses which ball to attempt, and the simulator uses the pretrained base and arm policies with the project's task-level pick-and-place logic. Add `--headless` to run without the PyBullet GUI (much faster).

**Example configuration:** `eval/example_config.json`

```json
{
  "balls": [[2.0, 1.5, 0.5], [-1.0, 3.0, 0.4], [3.5, -2.0, 0.6]],
  "zones": [[0.0, 4.0, 0.5], [-3.0, -1.0, 0.5], [4.0, 2.0, 0.5]],
  "assignments": [0, 1, 2]
}
```

*The output reports selected ball order and per-ball task results. A launched simulation does not by itself imply every placement succeeded; inspect the printed results.*

### Optional: training and hyperparameter tuning

**GPU training on Windows.** By default PyPI ships a CPU-only PyTorch for Windows. To use an NVIDIA GPU, add the CUDA wheel index to `RoboEnv/pixi.toml` and reinstall:

```toml
[feature.ml.pypi-options]
extra-index-urls = ["https://download.pytorch.org/whl/cu128"]
```

```bash
cd RoboEnv
pixi install -e ml
```

**Hyperparameter tuning** (`experiments/optuna_tune.py`) additionally needs Optuna:

```bash
cd RoboEnv
pixi add --feature ml --pypi optuna
```

## 🧪 Additional Workflows

The **main demo command above is the recommended starting point**. These separate commands are for specific experiments and do not need to be run first.

**Single-target learned reaching**

```bash
python eval/run_policy_sim.py --target 0.0 1.1 0.5
```

**Reference expert reaching**

```bash
python robotnik_arm_task.py
```

**Headless multi-target evaluation**

```bash
python eval/evaluate.py --pick_and_place --headless \
  --num_episodes 10 --num_targets_per_episode 3 --multi_target_seed 42
```

**Training your own models (optional).** The training entry points are `training/collect_data.py`, `training/split_dataset.py`, `training/train_regressor.py`, `training/collect_selector_data.py` and `training/train_selector.py`. They require generated `.npz` datasets and are **not required** when using the provided checkpoints. Review each script's `--help` options before training and retain the observation schema and normalisation metadata expected at inference.

## 🛠️ Troubleshooting

| Issue | Resolution |
|---|---|
| `ModuleNotFoundError: torch` | Activate `mlrai`; confirm `python` belongs to that environment. |
| `ModuleNotFoundError: simulation_and_control` | Install the compatible external RoboEnv distribution; it is not supplied by PyBullet. |
| `ModuleNotFoundError: pinocchio` | Install `pinocchio` from Conda Forge inside the active environment. |
| Missing `.pt` checkpoint | Run commands from `final/robotnik_arm_task/` and check the listed filenames. |
| URDF/configuration cannot be found | Confirm `models/` and `configs/` are present and compatible with the RoboEnv search paths. |
| GUI/display problem | Use a desktop graphics session or add `--headless` to supported evaluation/rollout commands. |
| Policy/feature mismatch | Use matching checkpoints and the normalisation/feature metadata saved during training. |

## ⚡ How It Works (TL;DR)

**Expert Robotnik control** generates state/action demonstrations → **behaviour cloning** trains base and arm policies → a separate **neural selector** learns the next-target ordering → **RoboEnv/PyBullet** runs closed-loop reaching and multi-ball tasks → **evaluation scripts** report results.

---

### Distribution note

Documentation and command examples were checked against the uploaded repository snapshot. Ensure the `main` branch retains the stated paths and checkpoints when publishing. The project also depends on the compatible external RoboEnv `simulation_and_control` package, which must be available to readers.
