#!/usr/bin/env bash
# chain_http_ab.sh   on spark4 in ~/work with </dev/null, detached: the chain-stop rule against the recipe's own rule
# through the live HTTP server, on qf_run.py's decode prompts (not the texts the rule was calibrated or chosen on).
# One image (production's); arms differ only by qwen-start.sh's environment: "new" = its defaults (MTP_DRAFTS 8,
# TENSORFOLD_CHAIN_RULE cum, TENSORFOLD_CHAIN_T 0.30), "old" = MTP_DRAFTS 6 + TENSORFOLD_CHAIN_RULE conf. 3 passes an
# arm, the order alternating each pass, every boot from a cold n-gram table (ple_evict.py, as abx_all.sh); pass 1 of
# each arm also runs golden. Then sum_ab.py (replies must be byte-identical across arms) and production restarted on
# the launcher's defaults. Log ~/work/chain-http-ab.log.
set -uo pipefail
cd "$HOME/work"
exec > chain-http-ab.log 2>&1
B=tensorfold-qwen38:v061-ots
rm -f chainab.jsonl
evict() { docker run --rm --entrypoint python3 -e HF_HUB_OFFLINE=1 -v /home/adam/.cache/huggingface:/root/.cache/huggingface \
  -v /home/adam/work:/work "$B" /work/ple_evict.py 2>&1 | grep ple_evict; }
boot() {   # arm
  (cd "$HOME/qwen38-flashnext" && ./stop.sh >/dev/null 2>&1) || true
  evict
  if [[ "$1" == old ]]; then MTP_DRAFTS=6 TENSORFOLD_CHAIN_RULE=conf bash "$HOME/qwen-start.sh" >/tmp/chainab-start.out 2>&1
  else bash "$HOME/qwen-start.sh" >/tmp/chainab-start.out 2>&1; fi
  for _ in $(seq 120); do curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8888/health | grep -q 200 && break; sleep 5; done
  echo "booted $1 $(date -Is): $(docker inspect qwen38-flash-next-tf --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^TENSORFOLD_CHAIN' | tr '\n' ' ') args: $(docker inspect qwen38-flash-next-tf --format '{{join .Args " "}}' | grep -oE -- '--mtp-drafts [0-9]+')"
}
for P in 1 2 3; do
  if [[ "$P" == 2 ]]; then ORDER="old new"; else ORDER="new old"; fi
  for ARM in $ORDER; do
    boot "$ARM"
    if [[ "$P" == 1 ]]; then
      python3 qf_harness.py golden "golden-chain-$ARM.json" | tail -1
      python3 qf_harness.py diff golden-pr319-final.json "golden-chain-$ARM.json" | tail -1
    fi
    RUNS=1 python3 qf_run.py "$ARM-p$P" chainab.jsonl decode > /dev/null
    echo "=== pass done $ARM p$P $(date -Is) ==="
  done
done
echo "=== A/B summary ==="
python3 sum_ab.py chainab.jsonl old new
echo "=== production on the launcher's defaults $(date -Is) ==="
boot new
echo "=== CHAIN HTTP AB DONE ==="
