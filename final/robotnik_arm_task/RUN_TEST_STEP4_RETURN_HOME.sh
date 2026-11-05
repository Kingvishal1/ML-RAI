#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Existing V2 script reads the latest checkpoint path and runs its original GUI command.
# New log name avoids overwriting the known-good release test.
bash RUN_TEST_RELEASE_V2.sh 2>&1 | tee v20_step4_return_home_test.log
