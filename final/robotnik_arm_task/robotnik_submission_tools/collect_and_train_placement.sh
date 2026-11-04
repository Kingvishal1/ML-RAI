#!/usr/bin/env bash
# Stop at the first failure; never train on a failed placement.
set -euo pipefail
bash robotnik_submission_tools/collect_expert.sh --target 2.50 3.00 1.0 --dropoff_xy 3.0 3.5
bash robotnik_submission_tools/collect_expert.sh --target 2.45 3.05 1.0 --dropoff_xy 3.0 3.5
bash robotnik_submission_tools/collect_expert.sh --target 2.50 3.00 1.0 --dropoff_xy 3.1 3.4
bash robotnik_submission_tools/collect_expert.sh --target 2.55 2.95 1.0 --dropoff_xy 3.1 3.4
python robotnik_submission_tools/train_placement_bc.py --demos expert_demos --output trained_placement_bc --epochs 300
