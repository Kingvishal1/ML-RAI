#!/usr/bin/env python3
"""Build a tidy, auditable Robotnik submission ZIP from actual Ubuntu project files.

Packaged candidate student weights must be trained from real NPZ demonstrations.
The ZIP truthfully records whether a CLOSED-LOOP student physical test passed.
No training episodes or unneeded experiment files are included in the ZIP.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

BASE = '060_bc_base_gelu_e70_h512_l3_16min_working.pt'
ARM = '051_bc_arm_gelu_e70_h512_l3_3min_working.pt'

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', default='.')
    p.add_argument('--candidate-dir', help='Output folder from REAL demonstration BC training')
    p.add_argument('--evaluation-json', help='Actual closed-loop student result JSON, if available')
    p.add_argument('--out', required=True, help='New output ZIP, refuses overwrite')
    a = p.parse_args()
    project = Path(a.project).resolve()
    robot = project/'robotnik_sensor_bc_v2'
    tools = project/'robotnik_submission_tools'
    files = {
        'src/robotnik_expert_controller.py': robot/'robotnik_expert_controller.py',
        'src/robotnik_bc_controller.py': robot/'robotnik_bc_controller.py',
        'src/candidate_bc_policy.py': robot/'candidate_bc_policy.py',
        'src/sensor_bc_v4.py': robot/'sensor_bc_v4.py',
        f'checkpoints/{BASE}': project/BASE,
        f'checkpoints/{ARM}': project/ARM,
        'checkpoints/grasp_sensor_residual_bc_v4.npz': robot/'grasp_sensor_residual_bc_v4.npz',
        'training/train_from_verified_expert.py': tools/'train_from_verified_expert.py',
        'training/check_candidate_policy.py': tools/'check_candidate_policy.py',
        'scripts/run_expert.sh': tools/'run_expert.sh',
        'scripts/run_student_bc.sh': tools/'run_student_bc.sh',
        'scripts/collect_expert.sh': tools/'collect_expert.sh',
        'README.md': tools/'README.md',
        'MODEL_STATUS.json': tools/'MODEL_STATUS.json',
    }
    candidate_dir = Path(a.candidate_dir).resolve() if a.candidate_dir else None
    if candidate_dir:
        for name in ('carry_candidate_bc.npz', 'release_candidate_bc.npz', 'training_report.json'):
            files['trained_students/'+name] = candidate_dir/name
    elif a.evaluation_json:
        p.error('--evaluation-json requires --candidate-dir')
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        p.error('Missing required file(s):\n  '+'\n  '.join(missing))
    verified = False
    evaluation = None
    if a.evaluation_json:
        evaluation = json.loads(Path(a.evaluation_json).read_text())
        match = (evaluation.get('carry_bc_sha256') == sha256(candidate_dir/'carry_candidate_bc.npz')
                 and evaluation.get('release_bc_sha256') == sha256(candidate_dir/'release_candidate_bc.npz'))
        verified = bool(match and evaluation.get('task_success')
                        and evaluation.get('full_closed_loop_student_test')
                        and evaluation.get('grasp_held_under_gravity')
                        and evaluation.get('carry_success')
                        and evaluation.get('release_success')
                        and evaluation.get('release_at_goal_success')
                        and evaluation.get('carry_mode') == 'bc'
                        and evaluation.get('release_mode') == 'bc')
        if not match:
            print('WARNING: Evaluation checkpoint hashes differ from trained files; verification invalid.')
        files['validation/student_closed_loop_result.json'] = Path(a.evaluation_json)
    out = Path(a.out)
    if out.exists():
        p.error(f'Refusing to overwrite {out}')
    manifest = {
        'project': 'Robotnik pretrained ML pickup/grasp + BC student carry and release actions',
        'success_definition': 'ball released at target under physical gravity; floor landing offset informational',
        'expert_baseline_included': True,
        'trained_student_weights_included': candidate_dir is not None,
        'student_closed_loop_physics_verified': verified,
        'student_control': 'BC predicts carry (v,omega) and release joint positions; high-level arrival state is threshold based',
        'training_provenance': 'Candidate training must use actual accepted expert NPZ traces, not logs or test weights',
        'files': {},
    }
    with zipfile.ZipFile(out, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=7) as z:
        for name,path in files.items():
            contents = path.read_bytes()
            z.writestr(f'robotnik_submission/{name}', contents)
            manifest['files'][name] = {'bytes': len(contents), 'sha256': hashlib.sha256(contents).hexdigest()}
        z.writestr('robotnik_submission/MANIFEST.json', json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:raise RuntimeError('Corrupt submission ZIP')
    print('BUILT:', out)
    print('BC CANDIDATE WEIGHTS:', 'INCLUDED' if candidate_dir else 'MISSING (expert/pretrained only)')
    print('BC STUDENT FULL PHYSICS:', 'PASS' if verified else 'NOT VERIFIED')

if __name__ == '__main__':
    main()
