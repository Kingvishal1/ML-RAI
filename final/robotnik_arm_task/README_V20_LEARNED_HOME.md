# V20 learned return to raised home — isolated experiment

This is a copy of the original V20 circular-drop controller with **only the return-home stage changed**. The original `robotnik_expert_controller.py`, V4 finger checkpoint, original circular zone code, and base/arm checkpoints are **not overwritten**.

After V20 verifies ball release, the existing goal-conditioned **arm BC model** gets the initial raised EE goal transformed into the robot's current base frame. Inference outputs 7 torque commands continuously. During learned return the original 18/2.5 arm joint-tracking servo is BYPASSED; there is no IK and no scripted motion trajectory. Arrival is checked using measured EE position and joint speed. **Exact initial 7-DOF joint angles are not guaranteed**, because the trained arm model is Cartesian-goal-conditioned, not joint-home-conditioned.

After an observed 0.4s stable arrival, the controller enters `ready_next_target`; the GUI keeps simulating and the learned arm model holds its goal. This is a future insertion point for sequential ball pickups; ball 2 and ball 3 are NOT yet implemented. Press **q** in the PyBullet GUI to finish.

**Important:** This is a new controller integration, not a newly trained policy, and has not been evaluated on the user's full PyBullet environment. If learned return cannot reach the raised goal it prints `V20_LEARNED_HOME_TIMEOUT` and reports no home success. Pickup, V4 ML grasp, V20 expert carry, expert lowering, and expert finger-opening are unchanged.

Install the ZIP from the root of `ML-RAI-affan-latest/final/robotnik_arm_task` with `unzip -o path/to/ZIP` then `bash RUN_V20_LEARNED_HOME_GUI.sh`. Run separately from the later multi-model project.
