# Audit of the uploaded Robotnik archive

The archive contains two runners, `grasp_reverse_bc.py`, `robotnik_arm_task.py`,
and `bc_demo_01.npz`. It does not contain base/arm weights, simulator configuration,
robot assets or the imported control helper modules.

Confirmed from source/data:

1. The dataset is 773 rows × 24 observations and 10 actions: only closing/reverse.
   Actions contain six finger targets and four wheel velocities, no arm torques.
2. Three input columns encode scripted phases; another input encodes elapsed close
   time and another displacement since the scripted reverse transition.
3. The old simulator freezes the sphere with mass=0, disables robot-ball collisions
   during approach, then attaches it using a fixed constraint. Base motion is also
   locked during arm approach. This cannot establish physical grasp success.
4. Arm command includes a joint-position/velocity holding correction. Navigation
   uses hand-coded switching/masking, and reverse termination uses base distance.
5. Recorded wheel commands range approximately from -0.7273 to zero rad/s. Earlier
   evaluation logs show +5 rad/s commands: those are outside demonstrated support.
   The precise causal contribution of normalization versus rollout drift cannot
   be established without running the old model and simulator.
6. Several observation dimensions vary very little in the single demonstration.
   Tiny standard-deviation normalization can amplify small runtime departures.
   The v2 model uses physically meaningful scale floors; this is a mitigation,
   not demonstrated closed-loop robustness.
7. The old validation randomly mixes neighbouring frames from the same trajectory.
   V2 uses separated chronological blocks but still has only one trajectory.
8. There are no examples of the final stopped state. V2 cannot claim learned stop.

Delivered changes are additive. The v2 neural network is genuinely fitted from
user data, but only the close/reverse subpolicy is trained here. The physical
runner retains explicitly documented hybrid control outside that subpolicy.
