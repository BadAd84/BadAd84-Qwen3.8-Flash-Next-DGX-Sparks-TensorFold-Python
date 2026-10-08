# mtp-graph - the MTP head's chain steps as CUDA graphs

**Status (2026-10-08): gates green, shipped with ple-defer on top.** Fresh interleaved HTTP A/B against production
(v061-ots 422254c9ef0b; `http_ab.sh ab1`, 3 passes, golden 6/6 on every arm, 8/8 replies identical; table in
`../ple-defer/README.md`): code C=1/2/4 90.2 / 63.6 / 42.1 -> 91.3 / 64.5 / 42.4 tok/s a request (+1.2 / +1.4 / +0.7%),
prose 74.6 / 51.0 / 32.2 -> 75.2 / 51.5 / 32.3 (+0.8 / +1.0 / +0.3%), prefill unchanged; with ple-defer
on top +1.8 to +2.2% in every cell. `mutants_mtpgraph.sh` exits 0: 7 of 7 mutants killed, none
hung. The two earlier survivors were the test's blind spots, not dead code: the test ran production's eager step
right before each replay on the same buffers, so the replay inherited everything the eager step had just written.
- `first` (`_pool_multi`'s block start) IS load-bearing: one row a stream at position p completes RATIO block p // 4
  when p % 4 == 3, and the replay must pool that block (the step's sparse select scores it at once). A stale `first`
  pools an old block instead and leaves the new one holding whatever was there. Killed now: the replay's written
  state (MTP cache row, scales, indexer key and the completed pooled block) is poisoned after the eager step and must
  come back bit-identical (`written state 5 differs` at context 44).
- `exact-context`: killed by the `1M` test (a ~1.4k-token prompt asked for a 1,100-token essay: one graph key from
  context 1,471 to 2,566, so replays cross the 512-token chunks at 1536 and 2048, the sparse onset at 2052 and the
  256-token score grid at 2304) with the attention scratch (chunk partials, key lists and counts, block scores)
  poisoned before each replay (`replay at context 1537 (captured at 1472): draft logits differ`).
- `multi-row-graphed` no longer hangs: the test fails a step of more than one row a stream that comes back from
  `ChainGraphs.run` as a graph at its capture (`a step of rows [3] ran as a graph`), any exception in a graph step
  ends the process with its FAIL line, and the runner time-boxes every run (twice the unmutated time plus 300 s; a
  run past it is a HANG, counted as a survivor, its container removed).

`tensorfold/families/qwen4_exp/cuda/multi.py` = chain-stop's `multi.py` plus `ChainGraphs`. A decode round drafts
with up to 8 MTP head steps; each step after the round's absorb (and an absorb of one kept row) steps one row per
chaining stream. Eager, a step is ~40 launches from Python whose host time is close to its GPU time, and the GPU
idles between steps for every host round trip (1 stream, in-process trace 2026-10-07: GPU idle 12% of steady decode;
per MTP step ~0.25-0.5 ms of gaps of 15-60 us before each `_embed`, `_rmsnorm`, `qmm`, `_qmm_hcdown`, `_qmm_upmix`).

`ChainGraphs.run` replays a CUDA graph of `mtp_compute` for such a step. Key: the stepping streams' MTP cache
addresses (the attention's captured pointer table) and a power-of-two context bucket (>= 8192). The attention's sparse
select is launched for every stream with the bucket as its context; its kernels read each row's real end from the
device (`_scores`/`_select` mask by `POS0`, `_chunks_multi`/`_merge_multi` by the row's position), so extra blocks and
chunks score, select and merge nothing and a replay computes the eager step's bits. Per replay only the rows'
positions are uploaded (one pinned copy of the step's int table). A key is captured the second time it is seen (a
one-off set of chaining streams stays eager); 32 graphs at most, the least recently replayed dropped.
Drafts change speed only; these drafts are also the eager drafts' bits, so acceptance is unchanged too.
`TENSORFOLD_MTP_GRAPHS=0` runs every step eagerly.

Measured (in-process, `inproc_ab.sh`, production's settings, steady-state decode, 3 passes x 2 windows an arm, the
arms interleaved; replies and per-stream acceptance identical in every window), tok/s a stream, medians (min-max):

| test | base | ple-native | ple-native + mtp-graph |
|---|---|---|---|
| 1 stream code | 85.5 (84.5-87.0) | 85.8 (+0.3%) | 87.5 (+2.3%) |
| 2 streams code | 60.0 (59.0-60.5) | 60.3 (+0.6%) | 60.9 (+1.6%) |
| 4 streams code | 41.6 (41.1-42.2) | 42.4 (+2.0%) | 42.7 (+2.8%) |
| 1 stream prose | 70.8 (68.9-73.0) | 70.9 (+0.2%) | 72.4 (+2.2%) |
| 2 streams prose | 51.3 (49.8-53.5) | 51.7 (+0.8%) | 52.3 (+2.1%) |
| 4 streams prose | 29.6 (29.5-29.7) | 29.9 (+0.9%) | 30.3 (+2.3%) |

An earlier version (capture at a key's first sight, all graphs dropped past 32) measured +1.4 / +2.2 / -1.0 / 0.0 /
+1.9 / +1.0% (4 streams code lost: 48 captures in a 400-token 4-stream run).

Files:
- `test_mtp_graph.py` (GPU, the serve stopped): before every replay the eager step (a fresh unbucketed table: v0.6.1's
  path) runs on the same state; its draft logits and the state it writes are kept, that state and the step's
  attention scratch are poisoned, and the replay's logits and written state must be bit-identical. Then the same
  requests with the graphs off, on with the checks, and on without them (production's path) must give identical
  replies, rounds and accepted drafts. Tests at 1, 2, 4 streams, `1M` (one key across a chunk step and a score-grid
  step; it FAILs unless replays crossed both) and `1L` (a ~7.9k prompt whose decode crosses the 8192 bucket). A test
  with no replay FAILs; a failed replay check or an exception in a graph step ends the process at once (exit 1).
  `MTPG_TESTS=1M,1L` runs only those tests (diagnosis). ple-defer's and hc-split's tests import it.
- `mutants_mtpgraph.sh <multi.py> <test_mtp_graph.py> [image]`: 7 mutants (stale positions, stale first positions,
  no table upload, exact instead of bucketed context, bucket missing from the key, multi-row steps graphed, graphs
  off), each run time-boxed. Not mutated (speed policy only, no bits): capture on second sight, the graph limit.

TensorFold's Apache-2.0 license applies as for `pr319/`.
