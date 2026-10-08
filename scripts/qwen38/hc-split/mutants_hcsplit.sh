#!/bin/bash
# mutants_hcsplit.sh <hc-split layer dir> <test_mtp_graph.py> [image]   on spark4 with the Qwen serve STOPPED (the
# test builds the whole engine; image: the stack hc-split ships on, its qmm.py replaced by the layer's): each mutant
# in mutate.py is written over a copy of the layer's qmm.py (every anchor must match EXACTLY once), test_hc_split.py
# runs against it in the image, and the mutant is killed only when the test exits non-zero AND prints a FAIL line.
# Every run is time-boxed: twice the unmutated run's time plus 300 s; a run past it is a HANG (its container removed)
# and a hang is a survivor. The unmutated layer must PASS first. Prints "exit N"; exit 0 only when every mutant is
# killed.
set -u
LAYER=$(readlink -f "$1") GRAPH_TEST=$(readlink -f "$2") IMAGE=${3:-tensorfold-qwen38:v061-ots}
CUDA=tensorfold/families/qwen4_exp/cuda
DST=/usr/local/lib/python3.12/dist-packages/$CUDA
W=$(mktemp -d -p "$HOME")
NAME="mutants-hcsplit-$$"
trap 'docker rm -f "$NAME" >/dev/null 2>&1; rm -rf "$W"' EXIT
LIMIT=7200
run() {   # directory holding qmm.py; sets rc (124: timed out) and secs
  local t0=$SECONDS
  timeout -k 20 "$LIMIT" docker run --rm --name "$NAME" --gpus all --ipc=host --ulimit memlock=-1 \
    -e HF_HUB_OFFLINE=1 -e TENSORFOLD_NO_UPDATE_CHECK=1 -e TENSORFOLD_PREFILL_ROWS=2048 -e TENSORFOLD_MTP_COPY=1 \
    -e TENSORFOLD_MEMORY_RESERVE_GIB=2 -e TENSORFOLD_VISION_WORKSPACE_MIB=0 -e TENSORFOLD_CHAIN_RULE=cum \
    -e TENSORFOLD_CHAIN_T=0.30 -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e TRITON_CACHE_DIR=/cache/triton \
    -v /home/adam/.cache/huggingface:/root/.cache/huggingface -v /home/adam/.cache/tensorfold-qwen38:/cache \
    -v "$1/qmm.py:$DST/qmm.py:ro" \
    -v "$LAYER/test_hc_split.py:/x.py:ro" -v "$GRAPH_TEST:/test_mtp_graph.py:ro" \
    --entrypoint python3 "$IMAGE" /x.py 400 > "$W/out" 2>&1
  rc=$?
  secs=$((SECONDS - t0))
  docker rm -f "$NAME" >/dev/null 2>&1
}
run "$LAYER/$CUDA"
if [ $rc -ne 0 ]; then
  echo "UNMUTATED FAILS (exit $rc):"; grep -aE "^FAIL|Error|Traceback" "$W/out" | head; echo "exit 1"; exit 1
fi
LIMIT=$((2 * secs + 300))
echo "unmutated: $(grep -ac '^PASS' "$W/out") PASS lines, $(grep -a '^exit' "$W/out"), ${secs} s (limit now ${LIMIT} s)"
grep -a "^PASS" "$W/out" | grep -aE "synthetic|up mixes"
survivors=0
for M in $(python3 "$LAYER/mutate.py" --list); do
  rm -rf "$W/m"
  if ! python3 "$LAYER/mutate.py" "$M" "$LAYER" "$W/m"; then
    echo "MUTANT $M: ANCHOR ERROR"; survivors=$((survivors + 1)); continue
  fi
  run "$W/m"
  fails=$(grep -ac '^FAIL' "$W/out")
  raised=$(grep -aq Traceback "$W/out" && echo ', raised')
  if [ $rc -eq 124 ] || [ $secs -ge $LIMIT ]; then
    echo "MUTANT $M: HUNG past ${LIMIT} s (exit $rc, $fails FAIL lines$raised): a survivor"
    survivors=$((survivors + 1))
  elif [ $rc -ne 0 ] && [ "$fails" -gt 0 ]; then
    echo "MUTANT $M: killed (exit $rc, $fails FAIL lines$raised, ${secs} s)"
    grep -a '^FAIL' "$W/out" | head -2 | cut -c1-240
  else
    echo "MUTANT $M: SURVIVED (exit $rc, $fails FAIL lines$raised, ${secs} s)"
    survivors=$((survivors + 1))
  fi
done
echo "survivors: $survivors"
echo "exit $((survivors > 0))"
exit $((survivors > 0))
