#!/bin/bash
set -eu
cd /home/user
bash redteam/run_all.sh > redteam/job.log 2>&1
