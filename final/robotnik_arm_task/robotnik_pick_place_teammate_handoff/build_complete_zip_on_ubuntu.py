#!/usr/bin/env python3
"""Create a SHAREABLE team handoff ZIP, including genuine checkpoints from Ubuntu.

This does not modify or delete project files. It deliberately ignores unrelated
logs, datasets, experiments and teammate files. Optional student weights MUST
come from the user's actual training output folder.
"""
import argparse, hashlib, json, zipfile, datetime
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('project',help='Path to original robotnik_arm_task project')
    ap.add_argument('--student-dir',help='Folder containing real carry_candidate_bc.npz and release_candidate_bc.npz')
    args=ap.parse_args()
    project=Path(args.project).expanduser().resolve()
    root=Path(__file__).resolve().parent
    required={
        'checkpoints/060_bc_base_gelu_e70_h512_l3_16min_working.pt':project/'060_bc_base_gelu_e70_h512_l3_16min_working.pt',
        'checkpoints/051_bc_arm_gelu_e70_h512_l3_3min_working.pt':project/'051_bc_arm_gelu_e70_h512_l3_3min_working.pt',
    }
    if args.student_dir:
        models=Path(args.student_dir).expanduser().resolve()
        for name in ['carry_candidate_bc.npz','release_candidate_bc.npz','training_report.json']:
            required['trained_students/'+name]=models/name
    missing=[str(v) for v in required.values() if not v.is_file()]
    if missing: ap.error('Required files not found:\n'+ '\n'.join(missing))
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dest=project/f'robotnik_pick_place_handoff_FULL_{stamp}.zip'
    if dest.exists(): ap.error('Output already exists: '+str(dest))
    inventory={}
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.name.endswith('.pyc') or '__pycache__' in p.parts:
                continue
            if p==Path(__file__).resolve() or p.name == 'HANDOFF_MANIFEST.json':
                continue
            arc='robotnik_pick_place_teammate_handoff/'+str(p.relative_to(root)).replace('\\','/')
            raw=p.read_bytes(); z.writestr(arc,raw)
            inventory[arc]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        for arc,path in required.items():
            raw=path.read_bytes()
            name='robotnik_pick_place_teammate_handoff/'+arc
            z.writestr(name,raw)
            inventory[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        status={'base_and_arm_checkpoints_included':True,
                'grasp_checkpoint_included':True,
                'trained_student_weights_included':bool(args.student_dir),
                'student_closed_loop_physics_verified':'NOT VERIFIED BY PACKAGER',
                'expert_ground_placement_verified':False,
                'task_success_definition':'grasp + carry + lower ball to near-floor pose + detached stable ground contact + final XY error <=0.05 m',
                'files':inventory}
        z.writestr('robotnik_pick_place_teammate_handoff/HANDOFF_MANIFEST.json',json.dumps(status,indent=2)+'\n')
    with zipfile.ZipFile(dest) as z: assert z.testzip() is None
    print('HANDOFF ZIP:',dest)
    print('Pretrained base/arm/grasp: INCLUDED')
    print('New student carry/release:', 'INCLUDED (training not independently verified)' if args.student_dir else 'NOT INCLUDED')
    print('No existing files deleted or changed.')
if __name__=='__main__':main()
