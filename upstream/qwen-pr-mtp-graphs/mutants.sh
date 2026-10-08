#!/bin/bash
# mutants.sh <recipe checkout with this PR applied> [image]   CPU only (--network none, 4 CPUs, 8 GB): the PR's check
# tools/mtp_graph_check.py must pass on the patched code, FAIL without this PR's patch, and FAIL on each mutant
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
PATCHES="0012-flash-next-copy-first.patch 0013-flash-next-ple-page-cache.patch 0015-flash-next-chain-stop.patch 0016-flash-next-mtp-graphs.patch"
CHECK=mtp_graph_check.py
W=$(mktemp -d -p "$HOME")
mkdir -p "$W/disk"
trap 'docker run --rm -v "$W:/w" --entrypoint rm "$IMAGE" -rf /w/disk >/dev/null 2>&1; rm -rf "$W"' EXIT
cat > "$W/mutate.py" <<'EOF'
import os, sys
MUTANTS = {
    'stale-positions': [('tensorfold/families/qwen4_exp/cuda/multi.py', 'vals[i] = vals[2 * k + i] = st.mtp_len', 'vals[2 * k + i] = st.mtp_len')],
    'stale-block-starts': [('tensorfold/families/qwen4_exp/cuda/multi.py', 'vals[i] = vals[2 * k + i] = st.mtp_len', 'vals[i] = st.mtp_len')],
    'table-not-uploaded': [('tensorfold/families/qwen4_exp/cuda/multi.py', '            ints.copy_(host, non_blocking=True)\n', '')],
    'exact-context': [('tensorfold/families/qwen4_exp/cuda/multi.py', 'bucket = max(GRAPH_BUCKET, 1 << (end - 1).bit_length())', 'bucket = end')],
    'bucket-not-in-key': [('tensorfold/families/qwen4_exp/cuda/multi.py', 'return (bucket, tuple((id(st),', 'return (0, tuple((id(st),')],
    'multi-row-graphed': [('tensorfold/families/qwen4_exp/cuda/multi.py', 'if any(a1 - a0 != 1 for _, a0, a1 in segs) or ', 'if ')],
    'image-graphed': [('tensorfold/families/qwen4_exp/cuda/multi.py', ' or any(st.image_positions is not None for st, _, _ in segs):', ':')],
    'captured-at-first-sight': [('tensorfold/families/qwen4_exp/cuda/multi.py', 'if self.seen[key] < 2:', 'if self.seen[key] < 1:')],
    'lru-not-refreshed': [('tensorfold/families/qwen4_exp/cuda/multi.py', '            self.graphs.move_to_end(key)\n', '')],
    'graphs-default-off': [('tensorfold/families/qwen4_exp/cuda/multi.py', 'MTP_GRAPHS = os.environ.get("TENSORFOLD_MTP_GRAPHS", "1") != "0"', 'MTP_GRAPHS = os.environ.get("TENSORFOLD_MTP_GRAPHS", "1") == "0"')],
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
for m in stale-positions stale-block-starts table-not-uploaded exact-context bucket-not-in-key multi-row-graphed image-graphed captured-at-first-sight lru-not-refreshed graphs-default-off; do run "$m"; done
echo "exit $bad"
exit $bad
