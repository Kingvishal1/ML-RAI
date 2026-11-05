"""NumPy-only checks. No claim of simulator or PyTorch execution."""
import ast
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
# Execute only the inspected pure-NumPy function definitions, no Torch dependencies.
tree=ast.parse((root/'train_full_policy.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gae')
ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'gae','exec'),ns)
a,r=ns['gae']([1,2],[0,0],[0,0],[0,1],[0,1],1,1)
np.testing.assert_allclose(a,[3,2])
a,r=ns['gae']([1,100],[2,0],[5,0],[0,1],[1,1],1,1)
np.testing.assert_allclose(a,[4,100]) # truncation bootstraps but does not leak next episode
code=(root/'full_policy.py').read_text();tree=ast.parse(code)
nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'SCALE','OBS_DIM'} for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name=='encode')]
ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),'encode','exec'),ns)
assert len(ns['SCALE'])==66
np.testing.assert_allclose(ns['encode'](np.zeros(66)),np.zeros(66))
x=np.zeros((2,66));x[1,11]=2*np.pi
np.testing.assert_allclose(ns['encode'](x)[0],ns['encode'](x)[1],atol=1e-6)
# Exercise command routing independently of the unavailable actual simulator.
from robotnik_env import RobotnikEnv
class P:
    WORLD_FRAME=2
    def getBasePositionAndOrientation(self,b):return ([2,2,1],[0,0,0,1])
    def applyExternalForce(self,*a):self.last=a
class Cmd:
    def SetControlCmd(self,values,modes):self.values=values;self.modes=modes
class Sim:
    def Step(self,cmd,mode):self.last=cmd
for sign in (-1.,0.,1.):
    e=RobotnikEnv.__new__(RobotnikEnv);e.low=np.array([-2]*4+[-20]*7+[0]*6,float)
    e.high=np.array([2]*4+[20]*7+[1.5]*6,float);e.mid=(e.low+e.high)/2;e.half=(e.high-e.low)/2
    e.targets=np.zeros(6);e.MotorCommands=Cmd;e.p=P();e.ball=1;e.mass=.05;e.sim=Sim();e.skip=2;e.dt=.001
    e.steps=0;e.max_steps=100;e.hold=0;e.prev_approach=1.;e.prev_goal=.25;e.prev_relative=np.zeros(3)
    info=dict(ee_ball_distance=1.,ball_goal_distance=.25,finger_groups=0,relative_ball=np.zeros(3),
              ball_speed=0.,base_speed=0.,joint_speed=0.,base_travel=0.,ball=np.array([2,2,1]),base=np.zeros(3))
    e.grip_base=None;e.retreat_direction=np.array([-1.,0.]);e.previous_base=np.zeros(3)
    e.sense=lambda:(np.zeros(66),info.copy())
    e.step(np.full(17,sign));expected=e.mid+sign*e.half
    np.testing.assert_allclose(e.sim.last.values[:11],expected[:11])
    np.testing.assert_allclose(e.targets,expected[11:])
    assert e.sim.last.modes==['velocity']*4+['torque']*13
for path in root.glob('*.py'):compile(path.read_text(),str(path),'exec')
print('PASS: syntax, observation shape/scaling, terminal/truncation GAE, direct 17-action routing (mock).')
print('NOT TESTED HERE: PyTorch training, RoboEnv integration, physical task success.')
