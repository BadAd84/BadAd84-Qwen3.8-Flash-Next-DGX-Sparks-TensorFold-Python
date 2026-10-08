#!/bin/bash
# mutants_mtpgraph.sh <patched multi.py> <test_mtp_graph.py> [image]   on spark4 with the Qwen serve STOPPED (the test
# builds the whole engine): each mutant of mtp-graph is applied to a copy of the patched multi.py (its anchor must
# match EXACTLY once), test_mtp_graph.py runs against it in the image, and the mutant is killed only when the test
# exits non-zero AND prints a FAIL line (a crash without a FAIL line is not a kill). Every run is time-boxed: twice the
# unmutated run's time plus 300 s; a run past it is a HANG, its container removed, and a hang is a survivor. The
# unmutated file must PASS first. Prints "exit N"; exit 0 only when every mutant is killed.
set -u
SRC=$1 TEST=$2 IMAGE=${3:-tensorfold-qwen38:v061-ots}
DST=/usr/local/lib/python3.12/dist-packages/tensorfold/families/qwen4_exp/cuda/multi.py
W=$(mktemp -d -p "$HOME")
NAME="mutants-mtpgraph-$$"
trap 'docker rm -f "$NAME" >/dev/null 2>&1; rm -rf "$W"' EXIT
LIMIT=7200
run() {   # multi.py to mount; sets rc (124: timed out) and secs
  local t0=$SECONDS
  timeout -k 20 "$LIMIT" docker run --rm --name "$NAME" --gpus all --ipc=host --ulimit memlock=-1 \
    -e HF_HUB_OFFLINE=1 -e TENSORFOLD_NO_UPDATE_CHECK=1 -e TENSORFOLD_PREFILL_ROWS=2048 -e TENSORFOLD_MTP_COPY=1 \
    -e TENSORFOLD_MEMORY_RESERVE_GIB=2 -e TENSORFOLD_VISION_WORKSPACE_MIB=0 -e TENSORFOLD_CHAIN_RULE=cum \
    -e TENSORFOLD_CHAIN_T=0.30 -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e TRITON_CACHE_DIR=/cache/triton \
    -v /home/adam/.cache/huggingface:/root/.cache/huggingface -v /home/adam/.cache/tensorfold-qwen38:/cache \
    -v "$1:$DST:ro" -v "$TEST:/x.py:ro" --entrypoint python3 "$IMAGE" /x.py 400 > "$W/out" 2>&1
  rc=$?
  secs=$((SECONDS - t0))
  docker rm -f "$NAME" >/dev/null 2>&1
}
run "$SRC"
if [ $rc -ne 0 ]; then
  echo "UNMUTATED FAILS (exit $rc):"; grep -aE "^FAIL|Error|Traceback" "$W/out" | head; echo "exit 1"; exit 1
fi
LIMIT=$((2 * secs + 300))
echo "unmutated: $(grep -ac '^PASS' "$W/out") PASS lines, $(grep -a '^exit' "$W/out"), ${secs} s (limit now ${LIMIT} s)"
grep -a "^PASS" "$W/out" | grep -a "replays"
survivors=0
mutant() {   # name, old, new (python escapes), applied once
  cp "$SRC" "$W/m.py"
  if ! python3 - "$W/m.py" "$2" "$3" <<'EOF'
import sys
p, old, new = sys.argv[1], sys.argv[2].encode().decode("unicode_escape"), sys.argv[3].encode().decode("unicode_escape")
s = open(p).read()
n = s.count(old)
if n != 1:
    print(f"anchor matched {n} times"); sys.exit(2)
open(p, "w").write(s.replace(old, new))
EOF
  then echo "MUTANT $1: ANCHOR ERROR"; survivors=$((survivors + 1)); return; fi
  run "$W/m.py"
  local fails raised
  fails=$(grep -ac '^FAIL' "$W/out")
  raised=$(grep -aq Traceback "$W/out" && echo ', raised')
  if [ $rc -eq 124 ] || [ $secs -ge $LIMIT ]; then
    echo "MUTANT $1: HUNG past ${LIMIT} s (exit $rc, $fails FAIL lines$raised): a survivor"
    survivors=$((survivors + 1))
  elif [ $rc -ne 0 ] && [ "$fails" -gt 0 ]; then
    echo "MUTANT $1: killed (exit $rc, $fails FAIL lines$raised, ${secs} s)"
    grep -a '^FAIL' "$W/out" | head -2 | cut -c1-240
  else
    echo "MUTANT $1: SURVIVED (exit $rc, $fails FAIL lines$raised, ${secs} s)"
    survivors=$((survivors + 1))
  fi
}
mutant stale-positions "vals[i] = vals[2 * k + i] = st.mtp_len" "vals[2 * k + i] = st.mtp_len"
mutant stale-first "vals[i] = vals[2 * k + i] = st.mtp_len" "vals[i] = st.mtp_len"
mutant no-upload "            ints.copy_(host, non_blocking=True)\n" ""
mutant exact-context "        step.ends = [bucket] * k\n" ""
mutant no-bucket-key "        return (bucket, tuple((id(st)," "        return (0, tuple((id(st),"
mutant multi-row-graphed "if any(a1 - a0 != 1 for _, a0, a1 in segs)" "if any(a1 - a0 > 99 for _, a0, a1 in segs)"
mutant no-graphs "MTP_GRAPHS = os.environ.get(\"TENSORFOLD_MTP_GRAPHS\", \"1\") != \"0\"" "MTP_GRAPHS = False"
echo "survivors: $survivors"
echo "exit $((survivors > 0))"
exit $((survivors > 0))
