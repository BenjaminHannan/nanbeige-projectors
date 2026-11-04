#!/bin/bash
# Mac side: every 20 min copy runs/ + logs/ (ONLY these: lead's cost rule 07:58) from both boxes into remote-sync/<box>/. Ends 11:20 ET.
D=~/learner-ops/premonition-projectors/remote-sync
while true; do
  for spec in "video 32262" "sound 30302"; do set -- $spec
    rsync -a --timeout=300 -e "ssh -p $2 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20" \
      --exclude='smoke_*/' --include='runs/***' --include='logs/***' --exclude='*' \
      root@ssh3.vast.ai:/workspace/proj/ $D/$1/ && echo "$(date +%T) synced $1" || echo "$(date +%T) SYNC FAIL $1"
  done
  [ $(date +%H%M) -ge 1120 ] && exit 0
  sleep 1200
done
