V20 SAME-WORLD 2-BALL REPEAT (EXPERIMENTAL)
======================================

DEFAULT (preserves existing pickup positions):
  Ball 1 pickup XYZ = 2.5 3.0 1.0
  Ball 2 pickup XYZ = 1.0 3.0 1.0
  SAME circular ground drop centre = 3.0 3.5 radius 0.50 m
  bash RUN_V20_REPEAT_TWO_BALLS.sh 2>&1 | tee v20_repeat_2ball.log

ALTERNATE SHORT-CARRY DEMO (different second pickup location; not a silent change):
  Ball 2 pickup XYZ = 3.5 3.0 1.0
  Same original circle centre and radius; ~0.71m carry instead of ~2.09m.
  bash RUN_V20_REPEAT_TWO_BALLS_SHORT_CARRY.sh 2>&1 | tee v20_repeat_short.log

WHAT "REFRESH" MEANS:
  Refresh the episode's policy observations, state machine, timers, physical
  contact counters, finger-open command and target. Do NOT reset the PyBullet
  physics world, existing ball 1, robot pose, or arm joints. Two physical balls
  exist in one simulation. This implementation already reinitializes run_sim()
  per ball while retaining a shared world. A known PyBullet motor-mode issue
  is cleared before second-ball learned arm inference (no pose reset).

WHAT IS ML / WHAT IS SCRIPTED:
  Existing trained base/7-DoF arm checkpoint + V4 neural finger model used
  for approach/grasp BOTH balls; ML arm torque cap bounded by URDF efforts.
  Grasp and carry monitored by physical contacts, gravity, and gripper-local slip.
  Carry navigation, Cartesian ground lowering, finger opening are the SAME V20
  expert/programmed controllers. This is HYBRID, NOT end-to-end learned.
  No virtual ball attachments, ball position hacks or teleports.

WHY THIS PATCH:
  The latest user run proved ball 2 physically reached the circle and expert
  fingers opened, but opening at a 0.300m offset and 0.985m height kicked the
  ball outside the circle (ground error 0.489m > allowed 0.470m).
  - Healthy 3-group grasp: original V20 near-ground lowering and expert opening
    on the second ball, just like ball 1.
  - Marginal grip (>=25mm local slip or fewer than three groups): continue
    physically driving toward the SAME circle centre. At 0.100m radius, if
    >=2 real finger groups still grip the ball and speed/force/offset gates
    pass, original expert opening is allowed. Never force a drop or claim
    success without ground-contact and inside-circle checks.

INSTALL in the original project directory:
  unzip -o ~/Downloads/V20_REPEAT_SAME_CIRCLE.zip
  python -m py_compile robotnik_sensor_bc_v2/robotnik_expert_controller_repeat_v20.py
  python VERIFY_REPEAT_V20.py

STATUS:
  Only source syntax and standalone gate tests verified by the assistant.
  Full PyBullet 2/2 physical success is UNVERIFIED here, and may still fail.
