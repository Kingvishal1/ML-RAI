"""Pure-math regression: rigidly held ball must have zero gripper-local slip."""
import numpy as np
pocket_offset = np.array([0.0, 0.015, -0.190])
ball_offset_from_pocket_in_hand = np.array([0.025, -0.012, 0.017])
hand_xyz = np.array([2.3, 2.7, 1.1])

def rotation_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

start_rot = rotation_z(0)
start_world_offset = start_rot @ ball_offset_from_pocket_in_hand
for degrees in (0, 30, 90, 135, 180, 270):
    rot=rotation_z(np.deg2rad(degrees))
    ball_world=hand_xyz + rot @ (pocket_offset + ball_offset_from_pocket_in_hand)
    pocket_world=hand_xyz + rot @ pocket_offset
    world_offset=ball_world - pocket_world
    # Same formula that the patched controller uses.
    current_local = rot.T @ (ball_world-hand_xyz) - pocket_offset
    reference_local = start_rot.T @ (hand_xyz + start_rot @ (pocket_offset + ball_offset_from_pocket_in_hand) - hand_xyz) - pocket_offset
    local_drift=np.linalg.norm(current_local-reference_local)
    world_drift=np.linalg.norm(world_offset-start_world_offset)
    assert local_drift < 1e-10, (degrees,local_drift)
    print(f'yaw_deg={degrees:3d} world-frame false slip={world_drift*1000:7.2f}mm, '
          f'gripper-frame measured slip={local_drift*1000:.6f}mm')
# A real 12mm displacement in hand frame must remain detectable.
rot=rotation_z(np.deg2rad(120))
ball_slipped=hand_xyz+rot @ (pocket_offset+ball_offset_from_pocket_in_hand+np.array([.012,0,0]))
measured=rot.T@(ball_slipped-hand_xyz)-pocket_offset
assert abs(np.linalg.norm(measured-reference_local)-0.012)<1e-10
print('PASS: rigid rotations rejected as false slip; actual local 12mm slip detected.')
