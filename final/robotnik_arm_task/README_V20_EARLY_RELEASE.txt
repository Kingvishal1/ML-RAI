V20 experimental two-ball physically verified in-zone early release fallback.

Observed from the user run (2026-11-04): ball 1 fully verified. Ball 2 grasp,
gravity hold, 2-m carry verified, but local ball slip reached 44.8 mm and only
two finger groups remained by circle arrival. During the first attempted lowering
motion (nearly zero commanded correction) ball 2 lost contact and was not delivered.

This variant leaves ball 1 V20 unchanged. The original arm ML approach, arm
motor re-enable, 40 Nm limited learned recovery, V4 six-finger ML predictions,
physical carry, gripper-local slip measurement, and expert finger-opening
policy remain exactly as before. It does NOT add a forced grasp or change
finger motion, friction, ball position, or touch the first-ball controller.

For ball 2 only, if ball is already within the circular-safe release region
and remains physically held by >=2 finger groups, moves <0.055 m/s, and has
local drift >=35 mm and <= original 85 mm abort threshold, it invokes the
same V20 expert finger-open immediately. This skips further wrist/IK lowering
for a fragile long-carry grasp. The ball must FALL NATURALLY from the current
height, detach from the entire robot, make non-robot ground contact, and rest
inside the circle before any delivery is counted. This is not identical
near-ground placement height; it is an explicit safety fallback.

No script auto-resets the arm or simulation between balls.

Installation from the V20 project directory:
  unzip -o ~/Downloads/V20_zone_safe_release.zip
  bash RUN_V20_TWO_BALLS_EARLY_RELEASE.sh 2>&1 | tee v20_zone_safe_release.log

Check:
  grep -E 'V20_SECOND_IN_ZONE_EARLY_RELEASE|V20_MULTI_DELIVERY_PASS|V20_DROP_ZONE_LANDING|V20_MULTI_FINAL|FAIL|ABORT' v20_zone_safe_release.log | tail -40

IMPORTANT: PyBullet is not installed in the assistant's environment.
Source passed Python compilation, and the release gate passed standalone
edge-case tests. A 2/2 simulated physical result is NOT yet established.
