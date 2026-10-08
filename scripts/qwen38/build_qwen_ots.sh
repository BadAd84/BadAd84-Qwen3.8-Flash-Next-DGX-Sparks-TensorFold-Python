#!/usr/bin/env bash
# build_qwen_ots.sh (spark4): tensorfold-qwen38:v061-ots = the MiaAI-Lab Qwen3.8 Flash Next recipe's image
# (tensorfold-qwen38:v0.6.1, recipe @ 4c0dea8) + seven OTS changes, all with golden replies byte-identical:
#  1. pr319/: TensorFold PR #319 ported to v0.6.1 (copy drafts proposed before the MTP chain, copy windows growing
#     16 -> 64 rows while copies land whole; verbatim repeats 153.8 -> ~535 tok/s, rewrites 141.4 -> ~322;
#     docs/SPEED-CAMPAIGN-2026-10.md section 2).
#  2. ple-cache/: the --ple-on-ssd n-gram lookups served from the page cache (a decode window's cached reads done on
#     the calling thread with RWF_NOWAIT; the next round's rows hinted with POSIX_FADV_WILLNEED as the drafts are
#     picked). Its multi.py is pr319's multi.py plus the hints, so it is copied AFTER pr319/ and wins.
#  3. ple-native/: those n-gram reads done in native threads with Python's GIL released (a load_inline C++ reader,
#     the same files, offsets and bytes; TENSORFOLD_PLE_NATIVE=0 restores the Python reads): prefill 8k +47%,
#     32k +12%. Its ssd_table.py is ple-cache's plus the native reader, so it is copied AFTER ple-cache/ and wins.
#  4. chain-stop/: where an MTP chain stops, by rule (TENSORFOLD_CHAIN_RULE; default "conf" = v0.6.1's). Its
#     multi.py is ple-cache's plus the rules; qwen-start.sh selects the measured rule.
#  5. mtp-graph/: the MTP head's one-row-a-stream chain steps replayed as CUDA graphs (TENSORFOLD_MTP_GRAPHS=0 runs
#     them eagerly). Its multi.py is chain-stop's plus ChainGraphs, so it is copied AFTER chain-stop/ and wins.
#  6. ple-defer/: a decode round's n-gram rows read when the forward reaches the n-gram layer, behind layers 0-1
#     (TENSORFOLD_PLE_DEFER=0 stages them first, as v0.6.1). Its multi.py is mtp-graph's plus the switch and its
#     forward.py the image's plus the deferred staging, so it is copied AFTER mtp-graph/ and wins. With 5:
#     decode +1.8 to +2.2% through HTTP.
#  7. tokenize-nogil/: cuda/server.py turns text into ids with HF tokenizers' encode_batch_fast (the same ids,
#     Python's GIL released), as GLM's patch 0082: encode held the GIL and stopped every other stream.
# Only the package files under each layer's tensorfold/ are COPY-ed over the image's site-packages (a layer's tests
# and mutant runners stay out); the base tag stays untouched.
# Usage: bash build_qwen_ots.sh [path to this directory on the box]
set -euo pipefail
HERE="${1:-$(dirname "$(readlink -f "$0")")}"
BASE=tensorfold-qwen38:v0.6.1
TAG=tensorfold-qwen38:v061-ots
SITE=/usr/local/lib/python3.12/dist-packages
docker image inspect "$BASE" >/dev/null || { echo "missing $BASE: run the recipe's ./start.sh once first"; exit 1; }
CTX=$(mktemp -d)
trap 'rm -rf "$CTX"' EXIT
mkdir -p "$CTX/files"
for PATCH in pr319 ple-cache ple-native chain-stop mtp-graph ple-defer tokenize-nogil; do   # in this order: a later copy of a file wins
  # CRs stripped: a context copied from a Windows checkout carried CRLF into the image (5 files of v061-ots, 2026-10-08)
  (cd "$HERE/$PATCH" && find . -type f -name '*.py' -path './tensorfold/*' | while read -r f; do mkdir -p "$CTX/files/$(dirname "$f")"; tr -d '\r' < "$f" > "$CTX/files/$f"; done)
done
printf '%s\n' "FROM $BASE" "COPY files/ $SITE/" 'LABEL tf.ots-patches="tensorfold-pr319-v061,ple-cache,ple-native,chain-stop,mtp-graph,ple-defer,tokenize-nogil"' > "$CTX/Dockerfile"
docker build -q -t "$TAG" "$CTX"
docker run --rm --entrypoint python3 "$TAG" -c "
import tensorfold.families.qwen4_exp.cuda.copies as c, tensorfold.families.qwen4_exp.cuda.multi as m
import tensorfold.families.qwen4_exp.ssd_table as t
assert hasattr(t.SSDTable, 'will_need') and hasattr(m.MultiDecoder, '_hint'), 'ple-cache files missing'
assert hasattr(t.SSDTable, '_gather_native'), 'ple-native files missing'
assert hasattr(m.MultiDecoder, '_chain'), 'chain-stop files missing'
assert hasattr(m, 'ChainGraphs'), 'mtp-graph files missing'
import tensorfold.families.qwen4_exp.cuda.forward as f
assert hasattr(m, 'PLE_DEFER') and hasattr(f, '_stage_ple'), 'ple-defer files missing'
import tensorfold.cuda.server as sv
assert hasattr(sv.App, '_encode_ids'), 'tokenize-nogil files missing'
print('built $TAG:', c.__file__, m.__file__, t.__file__)"
