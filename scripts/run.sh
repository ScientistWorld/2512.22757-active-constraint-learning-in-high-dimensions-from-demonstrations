#!/bin/bash
# End-to-end reproduction of the reference and the random-query control on
# both splits, from scratch: freezes their raw comparison records, publishes
# the reference-only scoring files and validates the workspace.
set -euo pipefail
cd /home/user
bash scripts/reproduce.sh
