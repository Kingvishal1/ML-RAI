from pathlib import Path
import ast

p = Path('robotnik_sensor_bc_v2/robotnik_expert_controller_two_ball_early_release.py')
tree = ast.parse(p.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
          and n.name == 'second_ball_safe_early_release')
namespace = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(p), 'exec'), namespace)
f = namespace['second_ball_safe_early_release']

class Circle:
    release_radius = 0.37
    def distance(self, xy):
        return ((xy[0]-3.0)**2+(xy[1]-3.5)**2)**0.5
    def can_release(self, xy):
        return self.distance(xy) <= self.release_radius

base = dict(ball_index=1, carry_success=True, drop_zone=Circle(),
    ball_xyz=[2.702,3.493,0.985], ball_speed_mps=0.026,
    contact_groups=[0,1], finger_normal_force_n=28.1,
    local_grip_slip_m=0.0448, max_grip_slip_m=0.085,
    arrival_margin_m=0.07)
assert f(**base), 'Actual failing ball-2 condition should choose early release'
for changed in [
    dict(ball_index=0), dict(contact_groups=[0]),
    dict(carry_success=False), dict(local_grip_slip_m=0.012),
    dict(local_grip_slip_m=0.100), dict(ball_speed_mps=0.15),
    dict(finger_normal_force_n=0.0), dict(ball_xyz=[2.5,3.5,0.985]),
    dict(ball_xyz=[2.69,3.5,0.985]),
]:
    args = {**base, **changed}
    assert not f(**args), f'Gate incorrectly triggered for {changed}'
print('V20_ZONE_SAFE_RELEASE_GATE_TESTS PASS (10 cases)')
