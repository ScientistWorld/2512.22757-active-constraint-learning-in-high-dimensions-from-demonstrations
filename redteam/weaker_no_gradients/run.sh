#!/bin/bash
set -eu
cd /home/user
python3 -m redteam.harness weaker_no_gradients
