# ple-defer - a decode round's n-gram rows read while the GPU runs layers 0-1

**Status (2026-10-08): gates green; ships on top of mtp-graph.** `test_ple_defer.py` exit 0 (17 PASS: every decode
round of every test deferred, 0 lost, 1,535 staged row sets bit-identical to the table, every reply / round count /
accepted count identical with the deferral off); `mutants_pledefer.sh` exit 0, 3 of 3 killed (`defer-never`: 0
deferred; `staging-dropped`: 86 lost, the replies change; `staging-late`: the replies change), none hung.

Interleaved HTTP A/B (`http_ab.sh ab1`, 3 passes, arm order rotated, cold n-gram table every boot; production
v061-ots 422254c9ef0b, mtp-graph alone 468b1d9ce2b6, mtp-graph + ple-defer 8e4a7bafb67d; golden 6/6 on every arm,
8/8 decode replies byte-identical across arms and passes), tok/s a request, median [pass medians]:

| workload | production | mtp-graph | mtp-graph + ple-defer |
|---|---|---|---|
| code C=1 | 90.2 [90.0, 90.2, 90.4] | 91.3 [91.1, 91.3, 91.4] (+1.2%) | 92.1 [91.7, 92.1, 92.7] (+2.1%) |
| code C=2 | 63.6 [63.0, 63.4, 64.0] | 64.5 [64.3, 64.6, 64.7] (+1.4%) | 64.9 [64.8, 65.0, 65.1] (+2.0%) |
| code C=4 | 42.1 [41.9, 42.1, 42.2] | 42.4 [42.2, 42.5, 42.6] (+0.7%) | 42.9 [42.7, 43.0, 43.1] (+1.9%) |
| prose C=1 | 74.6 [74.2, 74.6, 74.7] | 75.2 [75.0, 75.2, 75.8] (+0.8%) | 76.0 [75.6, 76.0, 76.1] (+1.9%) |
| prose C=2 | 51.0 [50.7, 50.7, 51.2] | 51.5 [51.3, 51.4, 51.6] (+1.0%) | 51.9 [51.9, 52.0, 52.0] (+1.8%) |
| prose C=4 | 32.2 [32.1, 32.1, 32.3] | 32.3 [32.0, 32.3, 32.4] (+0.3%) | 32.9 [32.6, 32.9, 33.0] (+2.2%) |
| prefill 8k / 32k / 128k | 2548 / 2634 / 2301 | 2543 / 2638 / 2298 | 2551 / 2640 / 2305 |

Every pass median of the ple-defer arm is above every pass median of production in all six decode cells.

`tensorfold/families/qwen4_exp/cuda/forward.py` = the image's `forward.py` plus `stage(..., defer_ple=True)`: the
token ids still go up in `stage`, but the n-gram (PLE) rows of every n-gram layer are looked up, read and copied when
the forward reaches its first n-gram layer (`ple_block` runs `b.ple_staging`), so the host reads them while the GPU
runs the layers before it. The same bytes land in the same buffers ahead of the same kernels on the stream; nothing
else changes. `multi.py` = mtp-graph's `multi.py` plus the switch: a decode round alone defers
(`TENSORFOLD_PLE_DEFER=0` stages them in `stage`, as v0.6.1); a round that shares its forward with a prompt pass
stages them in `stage` as before.

Files:
- `test_ple_defer.py` (GPU, the serve stopped; imports `../mtp-graph/test_mtp_graph.py`, mounted beside it): every
  decode round's staged rows read back from the GPU buffers and compared bit for bit with the table's rows of the
  ids `stage` looked up; a deferred staging the n-gram layer never ran is a FAIL; the deferral must engage; then the
  same requests with the deferral off and on must give identical replies, rounds and accepted drafts.
- `mutate.py` (the mutants, in Python so no shell quoting can corrupt an anchor) and `mutants_pledefer.sh <this dir>
  <test_mtp_graph.py> [image]`: each mutant time-boxed (a hang is a survivor); prints `exit N`.
  Mutants: the deferral never engaged, the staging dropped, the staging run after the n-gram kernels.
  Not mutated, with the reason: `b.staged.record()` after the deferred copies (every round already ends with a host
  sync on the same stream before the next `stage` writes the pinned buffers, so no test can observe it; kept as the
  event's documented meaning), the switch staying off beside a prompt pass (a speed choice: that round's forward is
  the pass's), and `b.ple_staging = None` in `ple_block` / the round's `finally` (they keep a deferred staging from
  running twice or outliving an exception: speed and exception paths, no bits).

TensorFold's Apache-2.0 license applies as for `pr319/`.
