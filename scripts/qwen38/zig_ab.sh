#!/usr/bin/env bash
# zig_ab.sh   on spark4 in ~/work with </dev/null, detached: the Qwen recipe's Zig release (1.0.0 in ~/qwen-zig, its
# own start.sh and defaults: TensorFold's Zig engine, the INT4-AutoRound checkpoint, FP8 KV, PARALLEL 8) against our
# production (qwen-start.sh: the Python recipe @ 4c0dea8 + the seven OTS layers), through the live HTTP server on
# qf_run.py's decode and prefill prompts. 3 passes an arm, the order alternating each pass; the Python arm boots from
# a cold n-gram table (ple_evict.py) as in stock_ab.sh. Pass 1 of each arm also records golden (not compared: the
# checkpoints differ). Then zig_sum.py and production restarted on qwen-start.sh. Log ~/work/zig-ab.log.
set -uo pipefail
cd "$HOME/work" || exit 1
exec > zig-ab.log 2>&1
B=tensorfold-qwen38:v061-ots
rm -f zigab.jsonl
evict() { docker run --rm --entrypoint python3 -e HF_HUB_OFFLINE=1 -v /home/adam/.cache/huggingface:/root/.cache/huggingface \
  -v /home/adam/work:/work "$B" /work/ple_evict.py 2>&1 | grep ple_evict; }
stopall() {
  (cd "$HOME/qwen-zig" && ./stop.sh >/dev/null 2>&1) || true
  (cd "$HOME/qwen38-flashnext" && ./stop.sh >/dev/null 2>&1) || true
  docker rm -f qwen38-flash-next-tf >/dev/null 2>&1 || true
}
boot() {   # arm
  stopall
  if [[ "$1" == zig ]]; then (cd "$HOME/qwen-zig" && ./start.sh </dev/null >/tmp/zigab-start.out 2>&1)
  else evict; bash "$HOME/qwen-start.sh" >/tmp/zigab-start.out 2>&1; fi
  for _ in $(seq 180); do curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8888/health | grep -q 200 && break; sleep 5; done
  echo "booted $1 $(date -Is): $(docker inspect qwen38-flash-next-tf --format '{{.Config.Image}}' 2>/dev/null)"
}
idle() { curl -s -m 3 http://127.0.0.1:8888/health |
  python3 -c 'import json,sys; h=json.load(sys.stdin); print(h.get("requests_running", h.get("running", 0)))' 2>/dev/null || echo 0; }
for _ in $(seq 240); do [[ "$(idle)" == 0 ]] && break; sleep 15; done
echo "=== start $(date -Is) ==="
for P in 1 2 3; do
  if [[ "$P" == 2 ]]; then ORDER="ots zig"; else ORDER="zig ots"; fi
  for ARM in $ORDER; do
    boot "$ARM"
    if [[ "$P" == 1 ]]; then python3 qf_harness.py golden "golden-zig-$ARM.json" | tail -1; fi
    RUNS=1 python3 qf_run.py "$ARM-p$P" zigab.jsonl > /dev/null
    echo "=== pass done $ARM p$P $(date -Is) ==="
  done
done
echo "=== A/B summary ==="
python3 zig_sum.py zigab.jsonl
echo "=== production on qwen-start.sh $(date -Is) ==="
boot ots
docker update --restart unless-stopped qwen38-flash-next-tf >/dev/null
echo "=== ZIG AB DONE ==="
