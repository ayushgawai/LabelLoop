#!/bin/zsh
# Waits for harness v9 to be finished (or committed), then launches the overnight sim training. Run detached:
#   nohup caffeinate -i ./scripts/queue_overnight.sh > results/queue_overnight.log 2>&1 &
cd "$(dirname "$0")/.."
deadline=$(( $(date +%s) + 4 * 3600 )); prev=""
log() { echo "[$(date +%H:%M:%S)] $*"; }
while true; do
  if git log -1 --format=%s | grep -qi "v9"; then log "v9 already committed"; break; fi
  if grep -q "reject_mode" configs/task_qr_sort.yaml 2>/dev/null && [ -n "$(git status --porcelain labelloop configs tests)" ]; then
    cur=$(git diff -- labelloop configs tests | shasum | cut -c1-16)
    if .venv/bin/python -m pytest -q -x > /tmp/ll_queue_pytest.txt 2>&1; then
      if [ "$cur" = "$prev" ]; then
        log "tests green twice with no changes -> committing v9"
        git add labelloop configs tests && git commit -q -m "Harness v9 (auto-committed by the overnight queue after the full suite passed twice, unchanged)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && break
      fi
      log "tests green (diff $cur), waiting for a stable second pass"; prev=$cur
    else
      log "tests not green yet ($(tail -1 /tmp/ll_queue_pytest.txt))"; prev=""
    fi
  else
    log "v9 not in the working tree yet"
  fi
  if [ $(date +%s) -ge $deadline ]; then log "deadline reached -> launching with current HEAD"; break; fi
  sleep 300
done
log "HEAD: $(git log -1 --format='%h %s')"
./scripts/go_overnight.sh && log "LAUNCHED" || log "launch FAILED"
