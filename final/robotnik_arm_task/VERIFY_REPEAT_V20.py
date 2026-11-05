from pathlib import Path
import ast
import math
src=Path('robotnik_sensor_bc_v2/robotnik_expert_controller_repeat_v20.py')
node=ast.parse(src.read_text())
function=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='second_ball_safe_early_release')
ns={}
exec(compile(ast.Module(body=[function],type_ignores=[]),str(src),'exec'),ns)
f=ns['second_ball_safe_early_release']
class Zone:
    release_radius=.37
    def distance(self,point): return math.hypot(point[0]-3.0,point[1]-3.5)
    def can_release(self,point): return self.distance(point)<=.37
z=Zone()
a=dict(ball_index=1,carry_success=True,drop_zone=z,ball_xyz=[2.70,3.493,0.985],
    ball_speed_mps=.026,contact_groups=[0,1],finger_normal_force_n=28,
    local_grip_slip_m=.0448,max_grip_slip_m=.085,center_radius_m=.10)
assert not f(**a), 'Observed prior bad 0.30m release must be prevented'
a['ball_xyz']=[2.92,3.49,.985]
assert f(**a), 'Center-safe measured physical release permitted'
for patch in [dict(contact_groups=[0]),dict(ball_index=0),dict(carry_success=False),
              dict(finger_normal_force_n=0),dict(local_grip_slip_m=.1),
              dict(local_grip_slip_m=.01),dict(ball_speed_mps=.15),
              dict(ball_xyz=[2.80,3.49,.985])]:
    assert not f(**(a|patch)),f'Incorrect early release: {patch}'
assert math.hypot(2.821-3,3.541-3.5)<.47
print('REPEAT V20: 10 measured-release gate tests passed; actual PyBullet run still required')
