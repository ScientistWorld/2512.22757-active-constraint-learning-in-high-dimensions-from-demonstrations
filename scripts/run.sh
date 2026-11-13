#!/bin/bash
# Development-only diagnostic pilot (no hidden-test data, nothing frozen).
# After the final hardening job, run.sh is restored to end-to-end
# reproduction (scripts/reproduce.sh).
set -euo pipefail
cd /home/user
bash scripts/pilot.sh
