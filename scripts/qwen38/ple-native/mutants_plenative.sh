#!/bin/bash
# mutants_plenative.sh <patched ssd_table.py> <test_ple_native.py> [image]   on spark4 (CPU and the checkpoint's
# n-gram files; it drops their pages, so run it with the serve stopped or idle): each mutant of ple-native is applied
# to a copy of the patched ssd_table.py (its anchor must match EXACTLY once), test_ple_native.py runs against it in
# the image, and the mutant is killed only when the test exits non-zero AND prints a FAIL line. The unmutated file
# must PASS first. Every mutant gets its own extension name, so the shared extension cache never serves another build.
# Prints "exit N"; exit 0 only when every mutant is killed.
set -u
SRC=$1 TEST=$2 IMAGE=${3:-tensorfold-qwen38:v061-ots}
DST=/usr/local/lib/python3.12/dist-packages/tensorfold/families/qwen4_exp/ssd_table.py
W=$(mktemp -d -p "$HOME")
# the containers write root-owned build files: remove them from a container, then the directory
trap 'docker run --rm -v "$W:/w" --entrypoint rm "$IMAGE" -rf /w/ext >/dev/null 2>&1; rm -rf "$W"' EXIT
run() { docker run --rm --cpus 16 -e HF_HUB_OFFLINE=1 -e TORCH_EXTENSIONS_DIR=/ext \
          -v /home/adam/.cache/huggingface:/root/.cache/huggingface:ro -v "$W/ext:/ext" \
          -v "$1:$DST:ro" -v "$TEST:/x.py:ro" --entrypoint python3 "$IMAGE" /x.py > "$W/out" 2>&1; }
mkdir -p "$W/ext"
run "$SRC"; rc=$?
if [ $rc -ne 0 ]; then echo "UNMUTATED FAILS (exit $rc):"; grep -aE "^FAIL|Error|Traceback" "$W/out" | head; echo "exit 1"; exit 1; fi
echo "unmutated: $(grep -ac '^PASS' "$W/out") PASS lines, $(grep -a '^exit' "$W/out")"
survivors=0
n=0
mutant() {   # name, old, new (python escapes), applied once
  n=$((n + 1))
  cp "$SRC" "$W/m.py"
  if ! python3 - "$W/m.py" "$2" "$3" "$n" <<'EOF'
import sys
p, old, new = sys.argv[1], sys.argv[2].encode().decode("unicode_escape"), sys.argv[3].encode().decode("unicode_escape")
s = open(p).read()
k = s.count(old)
if k != 1:
    print(f"anchor matched {k} times"); sys.exit(2)
s = s.replace(old, new).replace('name="tensorfold_ple_gather_ots"', f'name="tensorfold_ple_gather_mut{sys.argv[4]}"')
open(p, "w").write(s)
EOF
  then echo "MUTANT $1: ANCHOR ERROR"; survivors=$((survivors + 1)); return; fi
  run "$W/m.py"; local rc=$? fails
  fails=$(grep -ac '^FAIL' "$W/out")
  if [ $rc -ne 0 ] && [ "$fails" -gt 0 ]; then
    echo "MUTANT $1: killed (exit $rc, $fails FAIL lines$(grep -aq Traceback "$W/out" && echo ', raised'))"
    grep -a '^FAIL' "$W/out" | head -1
  else
    echo "MUTANT $1: SURVIVED (exit $rc, $fails FAIL lines$(grep -aq Traceback "$W/out" && echo ', raised'))"
    survivors=$((survivors + 1))
  fi
}
mutant parts-at-zero "at + first * width], axis=1)" "first * width], axis=1)"
mutant one-row-a-run "offsets[first], rows * width, at" "offsets[first], width + 0 * rows, at"
mutant parts-overlap "            base += n * width\n" ""
mutant skip-reads "const int64_t i = next.fetch_add(1);" "const int64_t i = next.fetch_add(2);"
mutant native-off "NATIVE = os.environ.get(\"TENSORFOLD_PLE_NATIVE\", \"1\") != \"0\"" "NATIVE = os.environ.get(\"TENSORFOLD_PLE_NATIVE\", \"1\") == \"0\""
echo "survivors: $survivors"
echo "exit $((survivors > 0))"
exit $((survivors > 0))
