#!/bin/bash
set -euo pipefail
cd /home/user
mkdir -p outputs
cat > outputs/gp_acl.json <<'JSON'
{"protocol":"acl-v2","command":["python","method/gp_acl.py","--variant","reference"]}
JSON
