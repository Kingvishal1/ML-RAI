#!/usr/bin/env bash
set -euo pipefail
bash RUN_V20_COLLECT_RELEASE_DEMOS.sh
bash RUN_V20_TRAIN_ONE_BALL_RELEASE.sh
bash RUN_V20_TEST_ONE_BALL_RELEASE.sh
