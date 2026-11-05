# V20: physically verified circular landing zone (expert control, ML grasp unchanged)

The drop-off argument is now a **centre** and `--drop_zone_radius` is the circle radius (default 0.50 m). A translucent green circle is drawn on the ground, with no collision/physics body. The ball must land, detach from the robot, remain in stable floor contact and finish **fully inside** the circle. The release gate keeps a 0.10 m margin plus the 0.03 m ball radius, so it opens only within 0.37 m of the centre by default. The released ball may drift naturally, but final success is checked with centre distance <= 0.47 m. No object teleports or artificial attachment.

The physical-carry phase now stops the base when the held ball reaches 0.30 m from the circle centre (0.37 m safe release radius minus 0.07 m approach margin); it does not drive all the way to the exact centre. The active lowering target follows the measured ball XY if inside this margin. It only corrects toward the nearest interior point if the ball moves outside. The former V19 repetitive reachability scans are skipped for this area-placement mode (while wrist/contact and speed safety guards remain active). Its learned V4 fingers and base/arm checkpoint files are unchanged. The expert controls the lowering, final finger-open action, and return-to-home.

Release is now at ball-centre height **0.55 m**, so it **drops** to the floor rather than doing close-contact ground placement; this is intentional to avoid the unstable low-Z IK posture shown in the log. Speed threshold defaults to 0.095 m/s and physical contact/settling checks still apply. Override via `--ground_ball_center_z 0.40 --ground_place_speed_max 0.04` for stricter but harder operation. The normal `run_expert.sh` launcher has these default values, custom CLI options appearing later override them.

To speed rendering: the script disables GUI panels and shadows when supported, removes artificial GUI sleep, prints fewer diagnostics, and samples closest-point gaps at 10 Hz instead of 50 Hz. Physics solver settings and simulation timestep are **not** reduced. You can also run `--headless` (PyBullet GUI disabled) for faster validation.

Install from project root:
```bash
cd ~/Desktop/UCL/ML/ML-RAI-affan-latest/final/robotnik_arm_task
conda activate comp0245
unzip -oq ~/Downloads/robotnik_pick_place_GROUND_PLACE_v20_CIRCULAR_DROP_ZONE.zip -d ~/Downloads/robotnik_v20
bash ~/Downloads/robotnik_v20/robotnik_pick_place_teammate_handoff/robotnik_bc_v9/install.sh "$PWD"
bash robotnik_submission_tools/run_expert.sh --target 2.5 3.0 1.0 --dropoff_xy 3.0 3.5 --drop_zone_radius 0.50 2>&1 | tee robotnik_v20_zone.log
grep -E "V20_|V9_GROUND_PLACE_POSE_VERIFIED|V8_RELEASE_RESULT|V19_ARM_RETURN|V9_GROUND_PLACEMENT_RESULT" robotnik_v20_zone.log | tail -70
```

**Full RoboEnv/PyBullet rollout requires the user's Ubuntu environment; controller-level tests and bytecode compilation do not verify success in simulation.**
