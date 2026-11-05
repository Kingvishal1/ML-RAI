#!/usr/bin/env python3
"""SAFE OPTIONAL CLEANUP. Dry-run by default; --apply MOVES obsolete scripts to archive.

Never touches checkpoints, dataset NPZ files, logs, training code, or V4.1.
"""
import argparse
from pathlib import Path
import shutil
from datetime import datetime

OLD_RUNNERS=(
  'run_grasp_carry_10cm_v5.py',
  'run_expert_carry_10cm_v5.py',
  'run_expert_carry_10cm_v5_REGENERATED.py',
  'run_expert_carry_100cm_v6.py',
  'run_expert_waypoint_release_v7.py',
  'run_ml_carry_to_3_3p5_v7.py',
  'run_expert_pick_carry_drop_v8.py',
  'run_expert_pick_carry_drop_v8_1.py',
  'run_expert_open_at_target_v8_2.py',
)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.',help='robotnik_arm_task root directory')
    p.add_argument('--apply',action='store_true',help='MOVE obsolete scripts to backup archive')
    p.add_argument('--delete',action='store_true',help='Permanently delete ONLY listed obsolete runner scripts')
    p.add_argument('--confirm-delete',action='store_true',help='Required together with --delete')
    a=p.parse_args()
    root=Path(a.project).expanduser().resolve();src=root/'robotnik_sensor_bc_v2'
    if not src.is_dir():p.error(f'Missing expected project subdirectory: {src}')
    if a.delete and (not a.confirm_delete or a.apply):
        p.error('--delete requires --confirm-delete and cannot be used with --apply')
    files=[src/name for name in OLD_RUNNERS if (src/name).is_file()]
    if not files:
        print('No obsolete V5–V8.2 runner scripts found. Nothing to archive.');return
    archive=root/('archive_old_robotnik_runners_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    for file in files:
        print(('DELETE' if a.delete else 'MOVE' if a.apply else 'DRY RUN')+f'  {file.name}'+('' if a.delete else f' -> {archive.name}/'))
    if a.delete:
        for file in files: file.unlink()
        print('Deleted only the listed obsolete runners. Models, data and final controller untouched.')
    elif a.apply:
        archive.mkdir(exist_ok=False)
        for file in files:shutil.move(str(file),str(archive/file.name))
        print('Preserved all old scripts in:',archive)
    else:
        print('No files changed. Re-run with --apply to archive them safely.')

if __name__=='__main__':main()
