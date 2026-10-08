#!/usr/bin/env bash
# zig2_ab.sh <arm> [passes=3]   on its own box in ~/work with </dev/null, detached: one arm of the two-box Qwen
# comparison (owner 2026-10-08). spark6 serves the recipe's Zig release (arm "zig", ~/qwen-zig started by its own
# start.sh, defaults); spark4 serves our production (arm "ots", qwen-start.sh, left serving as it is: no restart, the
# n-gram table warm). Both boxes start together and run qf_run.py's decode and prefill matrix against their own
# server (the client on the box, as stock_ab.sh), `passes` times; pass 1 also records golden replies (kept for reading,
# not compared: the checkpoints differ). /health's request counter is read around every pass, so a pass that saw
# requests it did not send is marked FOREIGN (production traffic inside the measurement). zig_sum.py summarizes the two
# arms' rows together. Log ~/work/zig2-<arm>.log, rows ~/work/zig2ab-<arm>.jsonl.
set -uo pipefail
ARM=$1
PASSES=${2:-3}
cd "$HOME/work" || exit 1
exec > "zig2-$ARM.log" 2>&1
rm -f "zig2ab-$ARM.jsonl"
health() { curl -s -m 5 http://127.0.0.1:8888/health; }
field() { python3 -c "import json,sys; print(json.load(sys.stdin).get('$1', -1))" 2>/dev/null || echo -1; }
idle() {
  for _ in $(seq 240); do
    r=$(health | field requests_running)
    [[ "$r" == 0 || "$r" == -1 ]] && return
    sleep 5
  done
}
echo "=== $ARM start $(date -Is) on $(hostname): $(docker inspect qwen38-flash-next-tf --format '{{.Config.Image}}')"
for P in $(seq "$PASSES"); do
  idle
  t0=$(health | field requests_total)
  sent=17
  if [[ "$P" == 1 ]]; then
    python3 qf_harness.py golden "golden-2box-$ARM.json" | tail -1
    sent=23
  fi
  RUNS=1 python3 qf_run.py "$ARM-p$P" "zig2ab-$ARM.jsonl"
  t1=$(health | field requests_total)
  if [[ "$t0" -ge 0 && "$t1" -ge 0 ]]; then
    extra=$(( t1 - t0 - sent ))
    note=""; [[ "$extra" -ne 0 ]] && note=" FOREIGN"
    echo "=== pass $P done $(date -Is): requests_total $t0 -> $t1, sent $sent, other $extra$note"
  else
    echo "=== pass $P done $(date -Is): no request counter in /health"
  fi
done
echo "=== ZIG2 $ARM DONE $(date -Is)"
