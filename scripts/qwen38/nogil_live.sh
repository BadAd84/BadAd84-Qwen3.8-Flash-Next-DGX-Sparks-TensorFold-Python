#!/usr/bin/env bash
# nogil_live.sh   on spark4 in ~/work with </dev/null, detached: put the tokenize-nogil layer into production. The
# current image is kept as tensorfold-qwen38:v061-ots-prenogil; build_qwen_ots.sh (from ~/qwen-ots, now with
# tokenize-nogil/) rebuilds v061-ots; qwen-start.sh restarts it; then golden (6/6 against golden-pr319-final.json)
# and one qf_run.py decode pass, whose replies must equal chain_http_ab.sh's "new" arm for the same prompts (the
# same chain rule; only tokenization changed). Any failure puts prenogil back. Log ~/work/nogil-live.log.
set -uo pipefail
cd "$HOME/work"
exec > nogil-live.log 2>&1
back() {
  echo "ROLLBACK: $1"
  docker tag tensorfold-qwen38:v061-ots-prenogil tensorfold-qwen38:v061-ots
  bash "$HOME/qwen-start.sh" restart >/dev/null 2>&1
  exit 1
}
docker tag tensorfold-qwen38:v061-ots tensorfold-qwen38:v061-ots-prenogil
bash "$HOME/qwen-ots/build_qwen_ots.sh" "$HOME/qwen-ots" || back "build"
docker image inspect -f '{{index .Config.Labels "tf.ots-patches"}}' tensorfold-qwen38:v061-ots
bash "$HOME/qwen-start.sh" restart >/tmp/nogil-start.out 2>&1 || back "start"
for _ in $(seq 120); do curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8888/health | grep -q 200 && break; sleep 5; done
python3 qf_harness.py golden golden-nogil.json | tail -1
python3 qf_harness.py diff golden-pr319-final.json golden-nogil.json | tail -1 | tee /tmp/nogil-golden
grep -q "6/6 identical" /tmp/nogil-golden || back "golden"
rm -f nogil-pass.jsonl
RUNS=1 python3 qf_run.py nogil-p1 nogil-pass.jsonl decode > /dev/null
python3 - <<'EOF' || back "replies differ from the chain A/B's new arm"
import json, sys
new = {}
for l in open("chainab.jsonl"):
    r = json.loads(l)
    if r["label"].startswith("new-") and r.get("sha"):
        new.setdefault((r["kind"], r["clients"], r["prompt"]), r["sha"])
rows = [json.loads(l) for l in open("nogil-pass.jsonl")]
same = sum(1 for r in rows if new.get((r["kind"], r["clients"], r["prompt"])) == r["sha"])
print(f"replies equal to the chain A/B's new arm: {same}/{len(rows)}")
sys.exit(0 if same == len(rows) and rows else 1)
EOF
python3 sum_ab.py nogil-pass.jsonl 2>/dev/null | head -14
echo "NOGIL LIVE DONE"
