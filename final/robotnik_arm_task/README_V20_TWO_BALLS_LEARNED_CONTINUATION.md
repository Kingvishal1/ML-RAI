# V20 Two-Ball Learned Continuation — Experimental

## Scope

A NEW Python entrypoint cloned from the prior two-ball V20 experiment; it does NOT overwrite either `robotnik_expert_controller.py` or `robotnik_expert_controller_two_ball.py`.

Both deliveries call the SAME `run_sim()` grasp/carry/drop state machine. The exact original V4 gripper policy, `.npz` checkpoint, contact-confirmed gravity activation, expert transport, expert opening and circular landing guard are unchanged. No welded/auto-attached ball and no robot/physics reset between deliveries. There is **no scripted return-home motion**.

## What changed

The old second cycle held the arm's end effector at its current, *lowered* post-place position throughout base navigation; then the BC arm stalled ~0.56m from ball 2.

While the base travels toward **ball 2**, and it is within 1.20m in the base drive phase, send a local goal toward the next ball to the original trained BC arm. When the base has reached its original switch threshold, the same arm policy continues with a capped relative target, then original V20 pocket calibration. This is **goal-conditioned learned action inference** with a deterministic target-scheduling guard; no joint PD/IK motion is used for this new transition.

If arm error stops improving for 7s (after the first 8s), stop honestly rather than looping ~80s. This is an unverified attempt to fix a model-distribution/reachability issue; it may still stall.

## Launch

Place the ZIP in Ubuntu Downloads. From the original V20 root:

```bash
conda activate comp0245
cd ~/Desktop/UCL/ML/ML-RAI-affan-latest/final/robotnik_arm_task
ZIP=$(find "$HOME" -type f -name 'robotnik_V20_TWO_BALLS_LEARNED_CONTINUATION.zip' -print -quit 2>/dev/null)
if [ -z "$ZIP" ]; then echo 'Download the ZIP into Ubuntu first'; else unzip -o "$ZIP"; fi
python -m py_compile robotnik_sensor_bc_v2/robotnik_expert_controller_two_ball_learned_continuation.py
bash RUN_V20_TWO_BALLS_LEARNED_CONTINUATION.sh 2>&1 | tee robotnik_v20_two_ball_continuation.log
```

## Expected markers

`V20_MULTI_DELIVERY_PASS ball=1` -> `V20_MULTI_LEARNED_ARM_PREVIEW ball=2` -> `V20_MULTI_LEARNED_ARM_FINAL ball=2` -> `V10 GRASP` -> `V4_GRAVITY_HOLD_PASS` -> `V20_MULTI_DELIVERY_PASS ball=2` -> `V20_MULTI_FINAL ... 2/2`.

If `V20_MULTI_ARM_STALLED` appears, the learned arm could not recover from the lowered post-placement configuration and we must change the learned policy/training or select a better base stopping pose. Do not assume that the existing V4 gripper or release failed if the arm never reached the ball.

## Verification status

Python syntax and static no-gripper-block-change comparisons passed in packaging environment. **PyBullet execution and physical success are not verified.**
