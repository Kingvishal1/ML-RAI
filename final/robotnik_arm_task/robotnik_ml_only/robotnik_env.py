"""RoboEnv adapter for the requested suspended-ball task.
All 17 motor targets come from the actor. Evaluation/reward never replace actions.
The dynamic ball receives constant gravity compensation: this is a weightless
object benchmark, NOT normal-gravity pickup. No table or virtual attachment.
"""
import os
import numpy as np

class RobotnikEnv:
    def __init__(self,headless=True,seconds=25,control_steps=20,seed=42,jitter=0.02):
        os.environ['ROBOENV_HEADLESS']='1' if headless else '0'
        from robotnik_arm_task import setup_simulation,setup_kinematics,FIXED_FINGER_ANGLES
        from robotnik_arm_control import configure_gripper_fingers,FINGER_JOINT_NAMES
        from simulation_and_control import MotorCommands
        import pinocchio as pin
        self.pin=pin;self.MotorCommands=MotorCommands
        self.sim,self.p,self.rid,cfg,root=setup_simulation()
        self.targets=configure_gripper_fingers(self.sim,self.p,self.rid,
                                              np.array(FIXED_FINGER_ANGLES,float,copy=True))
        self.dyn,self.ee_frame,_,_,_,_,_=setup_kinematics(self.sim,cfg,root)
        self.fingers=[j for j in range(self.p.getNumJoints(self.rid))
                      if self.p.getJointInfo(self.rid,j)[1].decode() in FINGER_JOINT_NAMES]
        active=list(self.sim.getNameActiveJoints())
        active=[n.decode() if isinstance(n,bytes) else str(n) for n in active]
        discovered=[self.p.getJointInfo(self.rid,j)[1].decode() for j in self.fingers]
        if len(active)!=17 or len(self.fingers)!=6 or active[11:17]!=discovered:
            raise ValueError(f'Unexpected active joint order; do not guess mapping: {active}')
        lo=[];hi=[]
        for j in self.fingers:
            info=self.p.getJointInfo(self.rid,j);l,h=float(info[8]),float(info[9])
            if not np.isfinite([l,h]).all() or h<=l:
                raise ValueError('Finger URDF limits must be finite and ordered')
            lo.append(l);hi.append(h)
        self.low=np.asarray([-2.272727]*4+[-20]*7+lo,dtype=np.float32)
        self.high=np.asarray([2.272727]*4+[20]*7+hi,dtype=np.float32)
        self.mid=(self.low+self.high)/2;self.half=(self.high-self.low)/2
        self.dt=float(self.sim.GetTimeStep());self.skip=int(control_steps)
        self.max_steps=int(seconds/(self.dt*self.skip))
        self.rng=np.random.default_rng(seed);self.jitter=jitter
        self.mass=.05
        self.ball=self.p.createMultiBody(baseMass=self.mass,
            baseCollisionShapeIndex=self.p.createCollisionShape(self.p.GEOM_SPHERE,radius=.03),
            baseVisualShapeIndex=self.p.createVisualShape(self.p.GEOM_SPHERE,radius=.03,
                                                         rgbaColor=[1,0,0,1]),
            basePosition=[2,2,1])
        self.p.changeDynamics(self.ball,-1,lateralFriction=1.2,
                               spinningFriction=.01,rollingFriction=.001,restitution=0)
        self.saved=self.p.saveState()
        self.reset()
    def reset(self):
        self.p.restoreState(self.saved)
        self.targets[:]=.1
        self.start_ball=np.array([2.,2.,1.])+self.rng.uniform(-self.jitter,self.jitter,3)
        self.p.resetBasePositionAndOrientation(self.ball,self.start_ball,[0,0,0,1])
        self.p.resetBaseVelocity(self.ball,[0,0,0],[0,0,0])
        self.sim.ReceiveObservation()
        start_base=np.asarray(self.sim.GetBasePosition())
        towards_start=start_base[:2]-self.start_ball[:2]
        towards_start/=max(np.linalg.norm(towards_start),1e-8)
        self.goal=self.start_ball.copy();self.goal[:2]+=.25*towards_start
        self.initial_base=start_base.copy();self.steps=0;self.hold=0.
        self.retreat_direction=towards_start.copy();self.grip_base=None;self.previous_base=start_base.copy()
        obs,info=self.sense()
        self.prev_approach=info['ee_ball_distance'];self.prev_goal=info['ball_goal_distance']
        self.prev_relative=info['relative_ball'].copy()
        return obs
    def sense(self):
        base=np.asarray(self.sim.GetBasePosition());quat=np.asarray(self.sim.GetBaseOrientation())
        yaw=float(self.p.getEulerFromQuaternion(quat)[2])
        q=np.asarray(self.sim.GetMotorAngles(0));qd=np.asarray(self.sim.GetMotorVelocities(0))
        lin=np.asarray(self.sim.GetBaseLinVelocity());ang=np.asarray(self.sim.GetBaseAngVelocity())
        qp=self.dyn.ReoderJoints2PinVec(np.concatenate([base,quat,q]),'pos')
        self.pin.forwardKinematics(self.dyn.pin_model,self.dyn.pin_data,qp)
        self.pin.updateFramePlacements(self.dyn.pin_model,self.dyn.pin_data)
        pose=self.dyn.pin_data.oMf[self.ee_frame]
        ee=pose.translation.copy();rotation=np.asarray(pose.rotation)
        ball=np.asarray(self.p.getBasePositionAndOrientation(self.ball)[0])
        bv=np.asarray(self.p.getBaseVelocity(self.ball)[0])
        contacts=self.p.getContactPoints(bodyA=self.rid,bodyB=self.ball)
        bits=np.array([float(any(c[3]==j and c[9]>0 for c in contacts)) for j in self.fingers])
        groups=sum(bool(bits[i:i+2].any()) for i in (0,2,4))
        obs=np.concatenate([base,[np.cos(yaw),np.sin(yaw)],lin,ang,q,qd,ee,ball,bv,
                            self.start_ball,bits,self.goal]).astype(np.float32)
        relative=rotation.T@(ball-ee)
        info=dict(ee_ball_distance=float(np.linalg.norm(ball-ee)),
                  ball_goal_distance=float(np.linalg.norm(ball-self.goal)),
                  finger_groups=groups,ball_speed=float(np.linalg.norm(bv)),
                  base_speed=float(np.linalg.norm(lin)),
                  joint_speed=float(np.max(np.abs(qd[4:11]))),
                  joint_speed_joint=int(np.argmax(np.abs(qd[4:11])) + 1),
                  joint_velocities=np.asarray(qd[4:11], dtype=float).copy(),
                  base_travel=float(np.linalg.norm(base-self.initial_base)),
                  relative_ball=relative,ball=ball,base=base)
        return obs,info
    def step(self,action):
        a=np.asarray(action,dtype=np.float32)
        if a.shape!=(17,) or not np.isfinite(a).all():
            raise ValueError('Actor must supply 17 finite commands')
        physical=self.mid+self.half*np.clip(a,-1,1)
        self.targets[:]=physical[11:17]
        cmd=self.MotorCommands()
        cmd.SetControlCmd(np.concatenate([physical[:11],np.zeros(6)]),
                          ['velocity']*4+['torque']*7+['torque']*6)
        for _ in range(self.skip):
            # Scene support only: constant upward force cancels 9.81m/s² gravity.
            # It has no goal, contact, distance, phase or policy-dependent switching.
            xyz=self.p.getBasePositionAndOrientation(self.ball)[0]
            self.p.applyExternalForce(self.ball,-1,[0,0,self.mass*9.81],xyz,self.p.WORLD_FRAME)
            self.sim.Step(cmd,'torque')
        self.steps+=1
        obs,info=self.sense();duration=self.skip*self.dt
        approach=info['ee_ball_distance'];goal=info['ball_goal_distance']
        grip=info['finger_groups']>=2 and approach<.12
        relative_speed=float(np.linalg.norm(info['relative_ball']-self.prev_relative)/duration)
        # Scripted task scoring is not scripted motor control.
        reward=4*(self.prev_approach-approach)+.02*np.exp(-10*approach)
        reward+=float(grip)*(.03+10*(self.prev_goal-goal))
        if grip and self.grip_base is None:
            self.grip_base=info['base'].copy()  # scoring reference only; never an action switch
        reverse_distance=(0. if self.grip_base is None else
            float(np.dot(info['base'][:2]-self.grip_base[:2],self.retreat_direction)))
        base_progress=float(np.dot(info['base'][:2]-self.previous_base[:2],self.retreat_direction))
        reward+=float(grip)*2*base_progress
        self.previous_base=info['base'].copy()
        reward-=.0002*float(np.square(a[:11]).sum())
        stationary=info['ball_speed']<.05 and info['base_speed']<.05 and info['joint_speed']<.3
        holding=grip and goal<.04 and reverse_distance>=.20 and relative_speed<.05 and stationary
        self.hold=self.hold+duration if holding else 0.
        success=self.hold>=1.
        safety_reasons=[]
        if not np.isfinite(obs).all(): safety_reasons.append('nonfinite_state')
        if info['joint_speed']>12: safety_reasons.append(
            f"joint_speed_gt_12:j{info['joint_speed_joint']}={info['joint_speed']:.2f}")
        if info['base_speed']>1: safety_reasons.append('base_speed_gt_1')
        if info['base_travel']>5: safety_reasons.append('base_travel_gt_5')
        if info['ball'][2]<.1: safety_reasons.append('ball_below_floor')
        safety=bool(safety_reasons)
        truncated=self.steps>=self.max_steps
        if success:reward+=10.
        if safety:reward-=5.
        self.prev_approach=approach;self.prev_goal=goal;self.prev_relative=info['relative_ball'].copy()
        info.update(success=success,safety_stop=safety,safety_reasons=safety_reasons,
                    truncated=truncated,hold_seconds=self.hold,relative_ball_speed=relative_speed,
                    reverse_distance=reverse_distance)
        return obs,float(reward),bool(success or safety),bool(truncated),info
    def close(self):
        self.p.disconnect()
