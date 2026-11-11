#!/bin/bash
# Final hardening job (see scripts/harden.sh). After it completes, run.sh is
# restored to the end-to-end reference reproduction (scripts/reproduce.sh).
set -euo pipefail
cd /home/user
bash scripts/harden.sh
