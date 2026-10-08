#!/bin/bash
# mutants.sh <recipe checkout with this PR applied> [image]   CPU only (--network none, 4 CPUs, 8 GB): the PR's check
# tools/copy_first_check.py must pass on the patched code, FAIL without this PR's patch, and FAIL on each mutant
# of the patched code. Each run starts a fresh container of the image and applies PATCHES (below, in order; the last
# is this PR's) from the checkout's patches/ with the recipe's own `patch -p0 --forward` in its site-packages, then the
# mutation:
# an anchor that does not match exactly once is an ERROR, never a pass. A kill needs the check's non-zero exit AND a
# FAIL line; a run past LIMIT seconds (default 900) is a HANG and counts as a survivor.
# Needs the checkpoint's config.json in the Hugging Face cache (HF_CACHE, default ~/.cache/huggingface).
# Prints one line a run and "exit N"; exit 0 only when the patched check passes, the unpatched one fails and every
# mutant is killed. Default image: the recipe's tensorfold-qwen38:v0.6.1.
set -u
R=$(readlink -f "${1:-.}")
IMAGE=${2:-tensorfold-qwen38:v0.6.1}
PATCHES="0012-flash-next-copy-first.patch"
CHECK=copy_first_check.py
W=$(mktemp -d -p "$HOME")
mkdir -p "$W/disk"
trap 'docker run --rm -v "$W:/w" --entrypoint rm "$IMAGE" -rf /w/disk >/dev/null 2>&1; rm -rf "$W"' EXIT
cat > "$W/mutate.py" <<'EOF'
import os, sys
MUTANTS = {
    'window-never-doubles': [('tensorfold/families/qwen4_exp/cuda/copies.py', 'next_copy_rows(self.rows, kept == verified, self.chain, self.most)', 'next_copy_rows(self.rows, kept != verified, self.chain, self.most)')],
    'proposal-fills-the-window': [('tensorfold/families/qwen4_exp/cuda/copies.py', 'min(self.rows - 1, left)', 'min(self.rows, left)')],
    'fit-newest-first': [('tensorfold/families/qwen4_exp/cuda/copies.py', 'sorted(live, key=lambda x: x.sid)', 'sorted(live, key=lambda x: -x.sid)')],
    'fit-drops-chain-rows': [('tensorfold/families/qwen4_exp/cuda/copies.py', 's.drafts = s.drafts[:chain - 1 + take]', 's.drafts = s.drafts[:take]')],
    'logits-not-compacted': [('tensorfold/families/qwen4_exp/cuda/multi.py', '            logits = logits[chains]\n', '            pass\n')],
    'copy-room-is-depth': [('tensorfold/families/qwen4_exp/cuda/multi.py', '            s.drafts = s.copies.propose(s.context, s.count - len(s.out)) if s.copies is not None else []\n        chains', '            s.drafts = s.copies.propose(s.context, self.depth) if s.copies is not None else []\n        chains')],
    'scratch-not-counted': [('tensorfold/cuda/geometry.py', 'linear * (scratch or each) *', 'linear * each *')],
    'copies-on-two-ranks': [('tensorfold/families/qwen4_exp/cuda/engine.py', 'copies = mtp and tp == 1', 'copies = mtp')],
    'drafts-before-take': [('tensorfold/families/qwen4_exp/cuda/multi.py', "        # after the take: a copy reads this round's tokens\n        self._draft_all([(s, a0, keep) for s, a0, keep, _, last in kept if s.draft and not last])\n", ''), ('tensorfold/families/qwen4_exp/cuda/multi.py', '        for s, _, _, new, _ in kept:\n', '        self._draft_all([(s, a0, keep) for s, a0, keep, _, last in kept if s.draft and not last])\n        for s, _, _, new, _ in kept:\n')],
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
    -v "$W/mutate.py:/m.py:ro" -v "${HF_CACHE:-$HOME/.cache/huggingface}:/hf:ro" -e MUTANT="$1" -e PATCHES="$PATCHES" --entrypoint bash "$IMAGE" -c '
    cd /usr/local/lib/python3.12/dist-packages || exit 9
    set -- $PATCHES; n=$#; i=0
    for p; do i=$((i + 1)); [ "$MUTANT" = stock ] && [ $i = $n ] && break
      patch -s -p0 --forward < /r/patches/$p || { echo "PATCH $p does not apply"; exit 9; }; done
    python3 /m.py || exit 8
    cd /tmp && python3 /r/tools/'"$CHECK"' $(ls -d /hf/hub/models--Vontra--Qwen3.8-Flash-Next-MLX-4bit-MTP/snapshots/*/ | head -1)config.json' > "$W/out" 2>&1
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
for m in window-never-doubles proposal-fills-the-window fit-newest-first fit-drops-chain-rows logits-not-compacted copy-room-is-depth scratch-not-counted copies-on-two-ranks drafts-before-take; do run "$m"; done
echo "exit $bad"
exit $bad
