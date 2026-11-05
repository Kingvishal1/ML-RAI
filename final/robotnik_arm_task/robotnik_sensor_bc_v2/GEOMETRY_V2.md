# Geometry V2 — experimental integration

Uses measured ball centre [0, 0.015, -0.190] metres in robot_j2s7s300_link_7 coordinates. Requires pocket error below 10 mm and no contact penetration deeper than 1 mm before closure. The ball becomes dynamic BEFORE closing. Finger servos use maximum velocity 0.5 rad/s and force 2 Nm (URDF limits: 1 rad/s, 2 Nm). No attachment or collision disabling.

The ball is 50g and gravity compensated after release. Before alignment it is static: this remains an artificial suspended-object setup. Approach switching, arm holding and servo limits remain deterministic; this is a hybrid runner, not a fully learned policy.

Validation: actual uploaded collision meshes tested in isolated fixed-hand PyBullet at 1/240s. A dynamic 30mm radius, 50g ball, initially at the stated centre, was closed on with a scripted 0.5 rad/s target ramp and 2Nm servos. After three seconds of closing and five further seconds, all three finger groups remained in contact; centre displacement was about 6.3mm. Repeated with gravity [0,0,0], [0,0,-9.81], [0,9.81,0], [9.81,0,0]. These are isolated feasibility checks, NOT learned-policy or mobile-robot success tests. Runner syntax compiles. Full RoboEnv integration and learned finger inference at this new pose remain untested.

Extract into the project root. Run your existing command, replacing run_grasp_only_air_v1.py with run_grasp_only_geometry_v2.py. The existing sensor_bc.py and grasp_reverse_sensor_bc_v2.npz are required. Startup must say GEOMETRY V2. If ALIGN stalls above 10mm, send the log; do not relax the alignment threshold to force closure. The earlier X server failure can be avoided for diagnostics by adding --headless.
