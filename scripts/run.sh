#!/bin/bash
# Final hardening job: checks, reference and random control rerun end to end
# on both splits (frozen and published), development and hidden-test audits.
set -euo pipefail
cd /home/user
bash scripts/harden.sh
