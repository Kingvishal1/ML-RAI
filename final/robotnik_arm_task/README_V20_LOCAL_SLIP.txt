V20 TWO-BALL PHYSICAL GRASP: LOCAL-FRAME CARRY SLIP CHECK (experimental)

Installation: unzip into the original V20 robotnik_arm_task directory. This
adds a NEW controller and launch script; it does NOT overwrite the original
V20 controller, V4 finger neural checkpoint, or the previous torque-fix code.

Run: bash RUN_V20_TWO_BALLS_LOCAL_SLIP.sh 2>&1 | tee robotnik_v20_local_slip.log

Same ball targets and shared circular drop zone as previous test. The learned
arm motor unlock and URDF-effort-bounded torque caps from the previous patch
remain, so ball 2's approach can reach. The original V4 finger predictions,
physical contact checks, 2-second gravity hold, and expert carry/lowering/
release policy remain unchanged.

FIX: The old carry slip monitor computed the difference of two BALL-POCKET
vectors in the WORLD frame. When the mobile robot rotates, even a rigidly held
ball changes this world vector, producing artificial 'slip'. This patch compares
ball-to-pocket offsets in the ROTATING HAND frame instead. It retains the SAME
85mm carry grip-loss threshold, SAME 2-finger-group contact requirement, and
SAME height-drop and circle-landing checks; the threshold is not weakened.
The corrected local-frame displacement is also used during reach-adjust and
lowering grip validation, for consistent real physical slip measurement.

Diagnostics: V20_CARRY_LOCAL_FRAME_ACTIVE, V20_FRAME_DIAG, V8_PHYSICAL_CARRY,
V20_MULTI_DELIVERY_PASS, V20_MULTI_FINAL. V20_FRAME_DIAG prints BOTH the old
world-frame (turn-induced) offset and new local-frame measured slip.

Caution: This is an experimental source-code correction. It has not been
validated end-to-end with PyBullet in the user's Ubuntu environment. Contacts
can still be genuinely lost. The controller must not mark success unless the
second ball is physically released and settled inside the same circle.
