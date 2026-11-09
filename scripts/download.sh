#!/bin/bash
# Estimated download: paper PDF only (<20 MB); no dataset or model weights.
# All procedural task generators and fixed raw comparisons ship in the repo.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/downloads paper
if [ ! -s paper/2512.22757.pdf ]; then
  curl --fail --location --retry 3 https://arxiv.org/pdf/2512.22757 -o paper/2512.22757.pdf
fi
