#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ -d v20_short_bend_demos ]] || { echo 'Missing 10g short bend demos' >&2; exit 2; }
[[ -d v20_release_v2_rollouts ]] || { echo 'No V2 correction rollouts collected' >&2;exit 2; }
# Never overwrite an old checkpoint.
OUT="trained_v20_release_v2_$(date +%Y%m%d_%H%M%S)"
python -u robotnik_submission_tools/train_v20_corrective.py \
 --demos v20_short_bend_demos --corrections v20_release_v2_rollouts \
 --out "$OUT" --epochs 180 --correction_weight 5 | tee v20_release_v2_train.log
[[ -s "$OUT/release_student.npz" ]] || exit 3
printf '%s\n' "$OUT" > v20_release_v2_checkpoint_dir.txt
echo "Saved new checkpoint in $OUT"
