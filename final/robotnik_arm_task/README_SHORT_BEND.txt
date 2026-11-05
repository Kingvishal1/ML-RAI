ONE BALL — SHORT BEND + NEURAL RELEASE TRAINING EXPERIMENT

Goal: ball (2.50,3.00,1.00), nearby drop zone centre (2.85,3.20), radius .32m.
The drop zone is INTENTIONALLY CHANGED from the original circle at (3.0,3.5)
so the arm/base requires less travel. This is a simpler, separate experiment.

INSTALL (from your Robotnik project root): unzip -o V20_SHORT_BEND_TRAIN.zip
RUN: bash RUN_TRAIN_SHORT_BEND.sh
TEST: bash RUN_TEST_SHORT_BEND.sh
TEST DIAG: grep -E 'V4_GRAVITY_HOLD_PASS|V4_HOLD_FAIL|V20_RELEASE_STUDENT_ACTION|V20_CIRCULAR_DROP_RESULT|FAIL' v20_short_bend_student.log | tail -40

Teacher demonstration collection is SCRIPTED. It is used only as supervision.
Learned runtime model predicts 7 arm targets, six finger targets and a release score.
The existing V4 model controls pickup/grasp. Carry navigation is STILL EXPERT
(SCRIPTED) in this version. Therefore this is NOT an entirely script-free robot.
The teacher also uses expert lowering as labels; do not conceal this provenance.

A successful student outcome is not assured; a teacher failure/unstable V4
physical grasp can prevent enough valid demonstrations. The physical grasp
checks should NEVER be disabled to make training pass.

The trained student is not handed scripted finger-opening commands in student
mode; a missing checkpoint causes an error. Release must be verified physically
in PyBullet with landing/contact checks. Do not claim success from training loss.

IMPORTANT: This package only addresses the short arm-and-release skill.
To remove ALL programmed motor actions, carry/navigation and phase transitions
require a different learned end-to-end policy with closed-loop training.
