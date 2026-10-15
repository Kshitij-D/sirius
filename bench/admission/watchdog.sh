#!/bin/bash
# Interrupt any benchmark srun step running longer than LIMIT seconds (hung run); log it.
LIMIT=${1:-1200}
while true; do
  for p in $(ps -u $USER -o pid=,etimes=,args= | awk -v L=$LIMIT '$2>L && $3=="srun" && /run2.sh/ {print $1}'); do
    echo "$(date +%F_%T) killing hung step pid $p: $(ps -o args= -p $p | cut -c1-160)" >> watchdog.log
    kill -INT $p; sleep 2; kill -INT $p 2>/dev/null
  done
  sleep 60
done
