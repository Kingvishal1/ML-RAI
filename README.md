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

1. Forms th
```bash
cd RoboEnv
pixi install -e ml
```

### 4. Activate the environment

Do this **in every new terminal** before running the project:

```bash
cd RoboEnv
