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

| Policy              | Responsibility                                     | Default checkpoint                              |
|---------------------|----------------------------------------------------|-------------------------------------------------|
| **Base BC**         | Navigation commands from robot/target observations | `final_navigation_policy.pt` |
| **Arm BC**          | Reaching control of the seven-joint arm            | `final_manipulation_policy.pt`   |
| **Target selector** | Next-ball choice from a multi-target scene         | `final_selector_policy.pt`             |

### 4. 🎯 Multi-Target Selection and Pick-and-Drop

`eval/run_policy_sim.py` implements the main rollout. In multi-target mode it reads `balls`, `zones`, and `assignments` from JSON (or generates a random scene), spawns physical ball objects and zone markers, and repeatedly:

1. Forms the current scene observation and chooses a remaining ball via the trained selector (or expert shortest-path ordering when selected).
2. Uses the base/arm controller to approach the selected ball and execute the task-specific pick-and-drop routine.
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
        │   ├── run_policy_sim.py       # Main rollout entry point
        │   ├── evaluate.py            # Repeated quantitative evaluation
        │   ├── compare_single.py      # Expert vs learned single-target comparison
        │   ├── compare_runs.py        # Expert vs learned PnP comparison
        │   ├── make_paper_charts.py   # Paper-ready chart generation
        │   ├── example_config.json    # Fixed multi-ball scene
        │   ├── parse_eval_log.py      # Evaluation-log processing
        │   ├── utils.py              # Evaluation helpers
        │   └── results/
        │       ├── final/            # Final evaluation results and charts
        │       └── archive/          # Older intermediate results
        ├── training/
        │   ├── collect_data.py        # Expert demonstration collection
        │   ├── split_dataset.py       # Base/arm dataset preparation
        │   ├── train_regressor.py     # Behaviour-cloning training
        │   ├── dagger_collect.py      # Expert-labelled policy-state data
        │   ├── merge_datasets.py      # Dataset merging
        │   ├── collect_selector_data.py # Selector training data
        │   └── train_selector.py      # Target-selection model training
        ├── sweep_results/             # Phase-switching threshold sweep data
        ├── experiments/               # Ablation, activation and scaling studies
        └── metrics/                   # Feature and threshold analyses
```


## 🚀 Getting Started

The **multi-target pick-and-drop demonstration** is the primary project entry point. The repository contains the required pretrained checkpoints, so retraining is not required for this demonstration.

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
git clone https://github.com/Kingvishal1/ML-RAI.git
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

### 5. ▶️ Single-Target Reaching

From `ML-RAI/final/robotnik_arm_task`, inside the pixi shell:

```bash
python eval/run_policy_sim.py --target 2 2 1.0
```

This drives the base towards the target at (2, 2, 1.0) m and then reaches with the arm. Add `--headless` to run without the GUI.

### 6. ▶️ Multi-Target Pick-and-Drop

From `ML-RAI/final/robotnik_arm_task`, inside the pixi shell:

```bash
python eval/run_policy_sim.py --pick_and_drop --config eval/example_config.json
```

Default checkpoints: `final_navigation_policy.pt` (base), `final_manipulation_policy.pt` (arm), and `final_selector_policy.pt` (selector).

The example scene configures **three balls, three drop zones and one assigned destination per ball**. During a multi-target episode, the neural selector chooses which ball to attempt, and the simulator uses the pretrained base and arm policies with the project's task-level pick-and-drop logic. Add `--headless` to run without the PyBullet GUI (much faster), especially for MacBook.

**Example configuration:** `eval/example_config.json`

```json
{
  "balls": [[2.0, 1.5, 0.5], [-1.0, 3.0, 0.4], [3.5, -2.0, 0.6]],
  "zones": [[0.0, 4.0, 0.5], [-3.0, -1.0, 0.5], [4.0, 2.0, 0.5]],
  "assignments": [0, 1, 2]
}
```
## 🏋️ Training Your Own Policies

Run from `ML-RAI/final/robotnik_arm_task` inside the pixi shell. Tuned hyperparameters are used by default.

**1. Collect expert demonstrations (base + arm)**
```bash
python training/collect_data.py --mode random --num_targets 1000 --xy_range 5 --vary_init --noise_std 1.0 --workers 8 --seed 42
```

**2. Split into base and arm datasets**
```bash
python training/split_dataset.py --data data/<dataset_file_name>.npz --out_dir data/split
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
python eval/run_policy_sim.py --pick_and_drop --config eval/example_config.json --base_checkpoint <base>.pt --arm_checkpoint <arm>.pt --selector_checkpoint <selector>.pt
```

## 📈 Results

Evaluation results from 200 single-target trials and 50 multi-object pick-and-drop episodes (3 balls each). All results are in `eval/results/final/`.

### Single-Target Reaching (200 targets, 5 mm tolerance)

| Metric | Expert | Learned |
|---|---|---|
| Reach rate (< 5 mm) | 6.0% | 85.0% |
| Mean best EE error | 4.84 mm | 4.29 mm |
| Median best EE error | 5.00 mm | 3.87 mm |
| Mean time | 17.0 s | 23.7 s |

The expert stops at a fixed 5 mm threshold, so nearly all trials cluster just outside tolerance. The learned policy converges closer on most targets but lacks the hard stopping rule, producing a wider error distribution.

### Out-of-Distribution Generalisation (50 targets outside training range)

| Metric | In-range (200) | OOD (50) |
|---|---|---|
| Reach rate (< 5 mm) | 85.0% | 76.0% |
| Mean best EE error | 4.29 mm | 3.93 mm |

Targets sampled at |x|, |y| in [5, 8] m, beyond the [-5, 5] m training range.

### Multi-Object Pick-and-Drop (50 episodes, 3 balls each)

| Metric | Expert | Learned |
|---|---|---|
| Grip rate | 90.0% | 99.3% |
| Delivery rate | 72.7% | 90.0% |
| Full-episode delivery | 36.0% | 70.0% |
| Mean time per episode | 145.8 s | 154.8 s |

The learned system maintains consistent performance across all three ball orders (96%, 86%, 88% delivery) while the expert degrades from 90% to 62% on later balls due to starting from non-nominal poses after each delivery.

## 📦 Pretrained Data

Pre-collected expert datasets are available for download if you want to skip the data-collection step:

- **Navigation and manipulation dataset:** [Download (.npz)](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucab434_ucl_ac_uk/IQCfNyR-AqJGQqZqI35xUlb2AVLqzDcsbDTsCVMASHXgOvI?e=IKx7Qi)
- **Selector dataset:** [Download (.npz)](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucab434_ucl_ac_uk/IQAuO1mXxGdASrmwbERiNiPNAfC8xLm5juV4UKSd3iViSOI?e=aW9jSp)

Place the downloaded files into `final/robotnik_arm_task/data/` and proceed from step 2 (split) in the training instructions above.

## 🔧 External Code Sources

- **RoboEnv** (`simulation_and_control`) — course-provided simulator, used unmodified
- **Standard libraries:** PyTorch, Pinocchio, PyBullet, NumPy, Matplotlib
- **Optuna** — used for hyperparameter tuning (learning rate, hidden size, number of layers, activation functions, dropout)

All policies, data pipelines, and evaluation code were written from scratch. No external pre-trained weights or checkpoints were used; all models were trained on self-collected expert demonstrations.
