#!/bin/bash
# Final hardening job: development-only noise-floor selection, controls from
# the final code on both splits, development and hidden-test audits. After
# this job, run.sh is restored to end-to-end reproduction (scripts/reproduce.sh).
set -euo pipefail
cd /home/user
bash scripts/harden.sh
