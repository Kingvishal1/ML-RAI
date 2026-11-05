# V21 learned return-home (separate BC policy)

New **neural** return-home policy: MLP trained on a labelled joint-servo teacher (behaviour cloning, not RL). The reference rule is used to make labels **during training only**. In `--return_home_controller bc` mode, every return-home arm setpoint comes from the network. PyBullet's physical position-control actuator and numerical safety bounds remain classical, not a scripted return-home path.

Run from the existing Robotnik project directory:

```bash
# Save your existing working script first
cp robotnik_sensor_bc_v2/robotnik_v20_one_ball_release_student.py robotnik_sensor_bc_v2/robotnik_v20_one_ball_release_student.py.before_v21
# Extract this patch ZIP into the project root, then:
bash RUN_TRAIN_RETURN_HOME_BC.sh
bash RUN_TEST_LEARNED_RETURN_HOME.sh
grep -E 'V21_LEARNED_RETURN_ACTION|V19_ARM_RETURN_PASS|V19_ARM_RETURN_TIMEOUT|V20_CIRCULAR_DROP_RESULT' v21_learned_return_home_test.log | tail -35
```

The separate model is saved at `trained_v21_return_home_bc/return_home_bc.npz`. Keep your existing V4 and V20 model checkpoints unchanged. No previously trained weights are overwritten. The main script's default is still `--return_home_controller expert` so your existing known-good release test command is unchanged. The V21 test script explicitly selects `bc`.

Important: the training dataset is **synthetic oracle-labelled states**, not verified simulator trajectories. This is a learned imitation of a classical feedback law; it is not reinforcement learning and does not by itself prove stable or collision-free return. Verify real simulated return-home physics before claiming success. If the actuator itself cannot return home, learning an oracle imitation will not fix the actuator; use the new per-joint debug logs to diagnose.

Optional: to collect real expert return trajectories, add `--return_home_record_npz path.npz` to an expert-return test. The recording is tagged as expert and whether return-home actually succeeded. The trainer only accepts successful examples via `bash RUN_TRAIN_RETURN_HOME_BC.sh --expert_npz path.npz`.
