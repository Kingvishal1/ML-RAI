"""Offline checks, not a replacement for Robotnik closed-loop validation."""
import argparse
import ast
from pathlib import Path
import numpy as np
from sensor_bc import SensorPolicy, convert_legacy

p=argparse.ArgumentParser()
p.add_argument('--data',required=True)
a=p.parse_args()
root=Path(__file__).resolve().parent
model=SensorPolicy(root/'grasp_reverse_sensor_bc_v2.npz')
with np.load(a.data,allow_pickle=False) as z:
    obs=z['obs']
x=convert_legacy(obs)
y=model.predict(x)
assert x.shape==(len(obs),17) and y.shape==(len(obs),10)
assert np.isfinite(y).all()
# The previous phase, timer and distance must have zero effect on this network.
altered=obs.copy(); altered[:,19:]=np.random.default_rng(7).normal(size=altered[:,19:].shape)*100
np.testing.assert_array_equal(convert_legacy(altered),x)
np.testing.assert_array_equal(model.predict(convert_legacy(altered)),y)
# Vectorized and single-observation loading/inference agree.
for i in (0,len(x)//2,len(x)-1):
    np.testing.assert_allclose(model.predict(x[i]),y[i],atol=1e-12)
assert np.std(y[:,0])>.05, 'Policy unexpectedly constant'
assert (y>=model.low-1e-12).all() and (y<=model.high+1e-12).all()
try:
    model.predict(np.full(17,np.nan))
except ValueError:
    pass
else:
    raise AssertionError('Nonfinite input accepted')
# Inspect executable calls rather than searching comments.
source=root/'run_grasp_reverse_contact_v2.py'
tree=ast.parse(source.read_text())
for node in ast.walk(tree):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
        assert node.func.attr not in {'createConstraint','setCollisionFilterPair',
                                      'resetBasePositionAndOrientation'}, node.func.attr
print('PASS: export inference, finite outputs, phase/timer/distance independence,')
print('      batching, action support bounds, invalid-input rejection, physics shortcut audit.')
print('NOT TESTED: Robotnik simulator integration, physical grasp, closed-loop success.')
