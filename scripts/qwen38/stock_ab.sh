#!/usr/bin/env bash
# stock_ab.sh   on spark4 in ~/work with </dev/null, detached: the recipe as shipped against production, through the
# live HTTP server on qf_run.py's decode and prefill prompts. "stock" = the recipe's own image
# (tensorfold-qwen38:v0.6.1, recipe @ 4c0dea8) started by its start.sh with its defaults (MTP 6, chain rule conf);
# "ots" = qwen-start.sh (tensorfold-qwen38:v061-ots: the seven OTS layers, MTP 8, chain rule cum 0.30). 3 passes an
# arm, the order alternating each pass, every boot from a cold n-gram table (ple_evict.py); pass 1 of each arm also
# runs golden, which must equal golden-pr319-final.json (the OTS layers change speed only). Then sum_ab.py (replies
# must be byte-identical across arms) and production restarted on qwen-start.sh. Log ~/work/stock-ab.log.
set -uo pipefail
cd "$HOME/work"
exec > stock-ab.log 2>&1
B=tensorfold-qwen38:v061-ots
rm -f stockab.jsonl
evict() { docker run --rm --entrypoint python3 -e HF_HUB_OFFLINE=1 -v /home/adam/.cache/huggingface:/root/.cache/huggingface \
  -v /home/adam/work:/work "$B" /work/ple_evict.py 2>&1 | grep ple_evict; }
boot() {   # arm
  (cd "$HOME/qwen38-flashnext" && ./stop.sh >/dev/null 2>&1) || true
  evict
  if [[ "$1" == stock ]]; then (cd "$HOME/qwen38-flashnext" && IMAGE=tensorfold-qwen38:v0.6.1 ./start.sh </dev/null >/tmp/stockab-start.out 2>&1)
  else bash "$HOME/qwen-start.sh" >/tmp/stockab-start.out 2>&1; fi
  for _ in $(seq 120); do curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8888/health | grep -q 200 && break; sleep 5; done
  echo "booted $1 $(date -Is): $(docker inspect qwen38-flash-next-tf --format '{{.Config.Image}}') $(docker inspect qwen38-flash-next-tf --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^TENSORFOLD_CHAIN' | tr '\n' ' ') args: $(docker inspect qwen38-flash-next-tf --format '{{join .Args " "}}' | grep -oE -- '--mtp-drafts [0-9]+')"
}
for P in 1 2 3; do
  if [[ "$P" == 2 ]]; then ORDER="ots stock"; else ORDER="stock ots"; fi
  for ARM in $ORDER; do
    boot "$ARM"
    if [[ "$P" == 1 ]]; then
      python3 qf_harness.py golden "golden-stock-$ARM.json" | tail -1
      python3 qf_harness.py diff golden-pr319-final.json "golden-stock-$ARM.json" | tail -1
    fi
    RUNS=1 python3 qf_run.py "$ARM-p$P" stockab.jsonl > /dev/null
    echo "=== pass done $ARM p$P $(date -Is) ==="
  done
done
echo "=== A/B summary ==="
python3 sum_ab.py stockab.jsonl stock ots
echo "=== production on qwen-start.sh $(date -Is) ==="
boot ots
docker update --restart unless-stopped qwen38-flash-next-tf >/dev/null
echo "=== STOCK AB DONE ==="
