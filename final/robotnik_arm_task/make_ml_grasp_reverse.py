#!/usr/bin/env python3
"""Make recording and learned-inference copies of the user's verified V10 script.

Never overwrites V10 or existing generated variants. Run from robotnik_arm_task,
with grasp_reverse_bc.py beside this generator. The supervised action network
imitates V10 gripper target positions + wheel velocities during close/reverse.
The original virtual ball constraint and scripted safety/state changes remain;
this is an incremental ML-controller milestone, not true contact-only grasp.
"""
from pathlib import Path
import ast
import re
import sys

src = Path('run_grasp_carry_v10_centered.py')
record = Path('run_grasp_reverse_ml_record.py')
evalfile = Path('run_grasp_reverse_ml_eval.py')
module = Path('grasp_reverse_bc.py')
if not src.exists():
    sys.exit('ERROR: verified V10 source is missing. Use the robotnik_arm_task directory.')
if not module.exists():
    sys.exit('ERROR: grasp_reverse_bc.py missing. Unzip the kit into the project directory.')
for dst in (record, evalfile):
    if dst.exists():
        sys.exit(f'ERROR: {dst} already exists. Files were not overwritten.')
source = src.read_text(encoding='utf-8')


def once(code, old, new, label):
    if code.count(old) != 1:
        raise RuntimeError(f'Expected one {label} anchor, got {code.count(old)}. Nothing written.')
    return code.replace(old, new, 1)


# argparse lives inside main(); inject one flag after the single-line parser
# statement. Don't depend on model loader / checkpoint implementation.
def add_arg(code, name, flag):
    pat = re.compile(r'(?m)^(?P<indent>\s*)parser\s*=\s*argparse\.ArgumentParser\([^\n]*\)\s*$')
    matches = list(pat.finditer(code))
    if len(matches) != 1:
        raise RuntimeError(f'Cannot locate argument parser for {name}; no changes made')
    m = matches[0]
    indent = m.group('indent')
    # Matching span is only parser definition, inserts next line.
    return (code[:m.end()] + '\n' + indent + flag + code[m.end():])


common_helper = '''
    # BC training features (same function used in recording and inference)
    from grasp_reverse_bc import features as grasp_bc_features
'''


def build_record():
    s = source
    s = add_arg(s, 'record_npz',
        'parser.add_argument("--record_npz", type=str, required=True, '
        'help="Output path for V10 expert-action demonstration .npz")')

    insert = '''
    # V10 teacher is left UNMODIFIED. Record only close/reverse examples.
    import atexit
    ml_obs, ml_action = [], []

    def save_demonstration():
        if not ml_obs:
            print("BC DATA ERROR: no closing/reverse observations recorded")
            return
        path = Path(args.record_npz)
        if path.exists():
            print(f"BC DATA ERROR: {path} already exists; refusing to overwrite")
            return
        np.savez_compressed(path,
            obs=np.asarray(ml_obs, dtype=np.float32),
            action=np.asarray(ml_action, dtype=np.float32))
        print(f"BC DATA SAVED: {path} ({len(ml_obs)} observations)")
    atexit.register(save_demonstration)
'''
    s = once(s, '    grasp_state = "approach"',
             '    grasp_state = "approach"' + insert,
             'grasp initialization for recorder')

    record_logic = '''        # Save teacher outputs once every 5 physics steps (~200 Hz at 1 kHz sim).
        # Only target FINGER angles and WHEEL velocities are supervised.
        if grasp_state in ("closing", "reverse") and step % 5 == 0:
            ball_xyz = p.getBasePositionAndOrientation(ball_id)[0]
            distance_reversed = (float(np.linalg.norm(base_pos[:2] - carry_start_xy))
                                 if carry_start_xy is not None else 0.0)
            close_elapsed = (max(0.0, current_time - grasp_start)
                             if grasp_start is not None else 0.0)
            ml_obs.append(grasp_bc_features(
                ee_pos, ball_xyz, q_mes[11:17], qd_mes[11:17],
                base_lin_vel[:2], base_bearing, distance_reversed,
                close_elapsed, grasp_state))
            ml_action.append(np.concatenate([
                np.asarray(target_finger_angles, dtype=float).reshape(6),
                np.asarray(wheel_cmds, dtype=float).reshape(4)]))

'''
    s = once(s, '        # ---- Send commands ----',
             record_logic + '        # ---- Send commands ----', 'send control anchor')
    s = s.replace('import argparse\n', 'import argparse\nfrom pathlib import Path\n', 1)
    ast.parse(s, filename=str(record))
    return s


def build_eval():
    s = source
    s = add_arg(s, 'grasp_checkpoint',
        'parser.add_argument("--grasp_checkpoint", type=str, '
        'default="grasp_reverse_bc.pt", '
        'help="Learned close/reverse controller checkpoint")')

    init = '''
    # Load learned action policy ONCE, without retraining base/arm.
    from grasp_reverse_bc import features as grasp_bc_features, load_controller
    grasp_bc_predict = load_controller(args.grasp_checkpoint)
    print(f"LEARNED GRIP+REVERSE: {args.grasp_checkpoint}")
'''
    s = once(s, '    grasp_state = "approach"',
             '    grasp_state = "approach"' + init,
             'grasp initialization for learned inference')

    # Never use the old fixed finger ramp as an actuator command in eval.
    finger_teacher = '            target_finger_angles[:] = finger_open + f * (finger_closed - finger_open)'
    s = once(s, finger_teacher,
             '            # Learned policy supplies six finger positions before sim.Step.',
             'remove scripted finger target')
    # Never use fixed -0.08 m/s wheel command as an actuator command in eval.
    reverse_teacher = '            wheel_cmds = np.full(4, -0.08 / WHEEL_RADIUS, dtype=float)'
    s = once(s, reverse_teacher,
             '            # Learned policy supplies four wheel velocities before sim.Step.',
             'remove scripted reverse velocity')

    policy_logic = '''        # ---- LEARNED GRIPPER + REVERSE MOTOR TARGETS ----
        # The V10 state machine still handles when to attach the *virtual*
        # ball constraint, when to unlock the base, and the 25 cm stop guard.
        # But six finger targets and reverse wheel speeds come from the network.
        if grasp_state in ("closing", "reverse"):
            ball_xyz = p.getBasePositionAndOrientation(ball_id)[0]
            distance_reversed = (float(np.linalg.norm(base_pos[:2] - carry_start_xy))
                                 if carry_start_xy is not None else 0.0)
            close_elapsed = (max(0.0, current_time - grasp_start)
                             if grasp_start is not None else 0.0)
            obs_ml = grasp_bc_features(
                ee_pos, ball_xyz, q_mes[11:17], qd_mes[11:17],
                base_lin_vel[:2], base_bearing, distance_reversed,
                close_elapsed, grasp_state)
            action_ml = grasp_bc_predict(obs_ml)
            if not np.isfinite(action_ml).all():
                raise RuntimeError("Learned gripper/reverse policy predicted nonfinite output")
            # Preserve the demonstrated open/closed range; do not invent
            # actuator targets outside the ranges in V10.
            finger_min = np.minimum(finger_open, finger_closed) - 0.03
            finger_max = np.maximum(finger_open, finger_closed) + 0.03
            target_finger_angles[:] = np.clip(action_ml[:6], finger_min, finger_max)
            if grasp_state == "reverse":
                wheel_cmds = np.clip(action_ml[6:10], -5.0, 5.0)
                if step % 500 == 0:
                    print("ML REVERSE wheel rad/s:", np.round(wheel_cmds, 2))
            elif step % 500 == 0:
                print("ML CLOSE finger targets:", np.round(target_finger_angles, 2))

'''
    s = once(s, '        # ---- Send commands ----',
             policy_logic + '        # ---- Send commands ----', 'send control anchor')
    ast.parse(s, filename=str(evalfile))
    return s


try:
    rec, ev = build_record(), build_eval()
except Exception as e:
    sys.exit(f'ERROR: {e}. Nothing written. V10 remains unchanged.')
record.write_text(rec, encoding='utf-8')
evalfile.write_text(ev, encoding='utf-8')
print('Created run_grasp_reverse_ml_record.py and run_grasp_reverse_ml_eval.py')
print('Original verified V10 and base/arm checkpoints were NOT modified.')
print('WARNING: the original V10 virtual ball grip is still present.')
