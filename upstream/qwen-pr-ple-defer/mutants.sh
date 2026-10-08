#!/bin/bash
# mutants.sh <recipe checkout with this PR applied> [image]   CPU only (--network none, 4 CPUs, 8 GB): the PR's check
# tools/ple_defer_check.py must pass on the patched code, FAIL without this PR's patch, and FAIL on each mutant
# of the patched code. Each run starts a fresh container of the image and applies PATCHES (below, in order; the last
# is this PR's) from the checkout's patches/ with the recipe's own `patch -p0 --forward` in its site-packages, then the
# mutation:
# an anchor that does not match exactly once is an ERROR, never a pass. A kill needs the check's non-zero exit AND a
# FAIL line; a run past LIMIT seconds (default 900) is a HANG and counts as a survivor.
# Prints one line a run and "exit N"; exit 0 only when the patched check passes, the unpatched one fails and every
# mutant is killed. Default image: the recipe's tensorfold-qwen38:v0.6.1.
set -u
R=$(readlink -f "${1:-.}")
IMAGE=${2:-tensorfold-qwen38:v0.6.1}
PATCHES="0012-flash-next-copy-first.patch 0013-flash-next-ple-page-cache.patch 0015-flash-next-chain-stop.patch 0016-flash-next-mtp-graphs.patch 0017-flash-next-ple-defer.patch"
CHECK=ple_defer_check.py
W=$(mktemp -d -p "$HOME")
mkdir -p "$W/disk"
trap 'docker run --rm -v "$W:/w" --entrypoint rm "$IMAGE" -rf /w/disk >/dev/null 2>&1; rm -rf "$W"' EXIT
cat > "$W/mutate.py" <<'EOF'
import os, sys
MUTANTS = {
    'defer-never': [('tensorfold/families/qwen4_exp/cuda/multi.py', 'defer_ple=PLE_DEFER and not (self.filling and self.converged)', 'defer_ple=False')],
    'defer-beside-a-pass': [('tensorfold/families/qwen4_exp/cuda/multi.py', 'defer_ple=PLE_DEFER and not (self.filling and self.converged)', 'defer_ple=PLE_DEFER')],
    'staging-dropped': [('tensorfold/families/qwen4_exp/cuda/forward.py', '        staging()\n', '')],
    'staging-late': [('tensorfold/families/qwen4_exp/cuda/forward.py', '        staging()\n', ''), ('tensorfold/families/qwen4_exp/cuda/forward.py', 'c.eps, c.streams, c.ngram_size)\n\n\ndef stage_ple_rows', 'c.eps, c.streams, c.ngram_size)\n    if staging is not None:\n        staging()\n\n\ndef stage_ple_rows')],
    'staging-not-cleared': [('tensorfold/families/qwen4_exp/cuda/multi.py', 'self.buf.gdn_tables = self.buf.attn_step = self.buf.ple_staging = None', 'self.buf.gdn_tables = self.buf.attn_step = None')],
    'defer-default-off': [('tensorfold/families/qwen4_exp/cuda/multi.py', 'PLE_DEFER = os.environ.get("TENSORFOLD_PLE_DEFER", "1") != "0"', 'PLE_DEFER = os.environ.get("TENSORFOLD_PLE_DEFER", "1") == "0"')],
}
name = os.environ["MUTANT"]
for path, old, new in MUTANTS.get(name, []):
    s = open(path).read()
    n = s.count(old)
    if n != 1:
        print(f"ANCHOR {name}: {path} matched {n} times: {old[:60]!r}")
        sys.exit(8)
    open(path, "w").write(s.replace(old, new))
EOF
bad=0
run() {   # mutant name ("" patched, "stock" without this PR's patch)
  CN=mutants-$$-$RANDOM
  timeout -k 10 "${LIMIT:-900}" docker run --rm --name "$CN" --network none -m 8g --cpus 4 -v "$R:/r:ro" \
    -v "$W/mutate.py:/m.py:ro"  -e MUTANT="$1" -e PATCHES="$PATCHES" --entrypoint bash "$IMAGE" -c '
    cd /usr/local/lib/python3.12/dist-packages || exit 9
    set -- $PATCHES; n=$#; i=0
    for p; do i=$((i + 1)); [ "$MUTANT" = stock ] && [ $i = $n ] && break
      patch -s -p0 --forward < /r/patches/$p || { echo "PATCH $p does not apply"; exit 9; }; done
    python3 /m.py || exit 8
    cd /tmp && python3 /r/tools/'"$CHECK"' ' > "$W/out" 2>&1
  rc=$?
  docker rm -f "$CN" >/dev/null 2>&1
  fl=$(grep -c "^FAIL" "$W/out")
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG $1 (past ${LIMIT:-900} s)"; bad=1
  elif grep -qE "^(ANCHOR|PATCH)" "$W/out"; then echo "ERROR $1: $(grep -E "^(ANCHOR|PATCH)" "$W/out")"; bad=1
  elif [ -z "$1" ]; then
    if [ $rc = 0 ] && grep -q "^ALL PASS" "$W/out"; then echo "patched: ALL PASS ($(grep -c "^PASS" "$W/out") checks)"
    else echo "patched: FAILS, exit $rc"; grep -E "^FAIL|Error" "$W/out" | head -5; bad=1; fi
  elif [ "$1" = stock ]; then
    if [ $rc != 0 ] && [ "$fl" -gt 0 ]; then echo "without ${PATCHES##* }: fails as it should ($fl checks FAIL)"
    else echo "without ${PATCHES##* }: does NOT fail (exit $rc, $fl FAIL lines)"; bad=1; fi
  elif [ $rc != 0 ] && [ "$fl" -gt 0 ]; then echo "KILLED $1 ($fl checks FAIL: $(grep -m1 "^FAIL" "$W/out" | cut -c6-90))"
  elif [ $rc = 0 ]; then echo "SURVIVED $1"; bad=1
  else echo "CRASHED $1, exit $rc without a FAIL line: $(grep -E "Error|error" "$W/out" | tail -1)"; bad=1; fi
}
run ""
run stock
for m in defer-never defer-beside-a-pass staging-dropped staging-late staging-not-cleared defer-default-off; do run "$m"; done
echo "exit $bad"
exit $bad
