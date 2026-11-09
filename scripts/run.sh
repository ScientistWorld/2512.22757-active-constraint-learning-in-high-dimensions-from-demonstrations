#!/bin/bash
set -eu
cd /home/user
bash redteam/run_followup.sh > redteam/followup_job.log 2>&1
