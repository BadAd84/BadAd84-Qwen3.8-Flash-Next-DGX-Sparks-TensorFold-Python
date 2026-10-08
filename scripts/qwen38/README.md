# scripts/qwen38 - the Qwen3.8 Flash Next serve on spark4

spark4 serves Qwen3.8 Flash Next (`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`) on `:8888` with the MiaAI-Lab recipe
`Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold` (@ 4c0dea8, clone `~/qwen38-flashnext` on the box) and seven OTS
changes: TensorFold PR #319 ported to the recipe's TensorFold v0.6.1, `ple-cache`, `ple-native`, `chain-stop`,
`mtp-graph`, `ple-defer` and `tokenize-nogil` (below). Measurements and verdicts:
`docs/SPEED-CAMPAIGN-2026-10.md` section 2. Each layer directory holds its files (under `tensorfold/`) and, from
`ple-native` on, its test, mutant runner and a README with the evidence.

- `qwen-start.sh` (box copy `~/qwen-start.sh`): start or restart the serve with the OTS image and the measured MTP
  chain settings (`MTP_DRAFTS=8`, `TENSORFOLD_CHAIN_RULE=cum`, `TENSORFOLD_CHAIN_T=0.30`; each overridable from the
  environment). A plain `./start.sh` in the recipe would start the recipe's own image and drop every OTS change.
- `build_qwen_ots.sh`: builds `tensorfold-qwen38:v061-ots` = the recipe's `tensorfold-qwen38:v0.6.1` + the package
  files (`tensorfold/...`) of `pr319/`, `ple-cache/`, `ple-native/`, `chain-stop/`, `mtp-graph/`, `ple-defer/`,
  `tokenize-nogil/` in that order, copied over its site-packages (a later directory's copy of a file wins).
- `pr319/`: six TensorFold v0.6.1 source files (`tensorfold/cuda/geometry.py`, `tensorfold/families/qwen4_exp/cuda/
  {__init__,copies,decode,engine,multi}.py`) as modified by porting **TensorFold PR #319** ("copies from the context
  before MTP chains, windows growing to 64 rows", by Arminova; closed unmerged when TensorFold's Python engine froze)
  from TensorFold 0.6.4 to 0.6.1. TensorFold (github.com/ashhart/TensorFold) is Apache-2.0; these files keep its
  license, and the changes relative to v0.6.1 are exactly that port (`copies.py` is new; the others gain the
  copy-first drafting, the growing copy window and the row budget).
- `ple-cache/`: two files (`tensorfold/families/qwen4_exp/ssd_table.py` from v0.6.1, `.../cuda/multi.py` from
  `pr319/`) changed so the `--ple-on-ssd` n-gram lookups of decode rounds come from the page cache instead of a
  16-thread pread storm per stream per round. Measured cause (2026-10-07, in-process spans + Kineto): the round's
  `stage()` gathers left the GPU idle 3.6 / 8.0 / 10.0% of decode wall at 1 / 2 / 4 streams (prose), and a gather
  costs ~1.3 ms per window even with every page cached, because 16 threads hand ~9 us each of GIL work per 0.4 us
  read. The change: (1) `SSDTable.gather` does a decode-sized gather's (<= 4096 reads) cached reads on the calling
  thread with `RWF_NOWAIT` and hands only the misses to the threads; (2) `MultiDecoder` hints the next round's rows
  (its first token after the accept, each copy, then each draft as the chain picks it) with
  `POSIX_FADV_WILLNEED` on a one-thread executor while the GPU runs the MTP steps. Both read the same file bytes
  (gather output compared equal on 1,280 windows cold and warm; golden 6/6 and every A/B reply byte-identical).
  Interleaved HTTP A/B, 3 passes an arm, medians (per-request tok/s): prose C=1/2/4 66.2 / 45.6 / 28.4 ->
  73.9 / 50.1 / 31.9, code 77.7 / 58.8 / 36.9 -> 85.1 / 64.0 / 41.0 (+9 to +12%, pass spreads disjoint).
  Off switches (the recipe's start.sh passes only `TENSORFOLD_*` variables into the container):
  `TENSORFOLD_PLE_NOWAIT=0` (every read to the threads, as v0.6.1), `TENSORFOLD_PLE_HINTS=0` (no hint thread).
  TensorFold's Apache-2.0 license applies as for `pr319/`.
- `chain-stop/`: `.../cuda/multi.py` (ple-cache's plus chain rules) choosing where a round's MTP chain stops; drafts
  change speed only, so every rule keeps the reply's bits. `TENSORFOLD_CHAIN_RULE`: `conf` (default: v0.6.1's, a
  draft under `--mtp-confidence` ends the chain), `cum` (the chance that every draft so far lands, the product of
  their calibrated draft-head probabilities, falls under `TENSORFOLD_CHAIN_T`), `cost` (keep a draft while its
  chance buys more than a verify row's ~3.1 ms at 1 stream / ~2.2 ms at 4, chain on while the next pays an MTP
  step's ~1.4 ms too), `adapt` / `acum` (`cost` / `cum` scaled by each stream's recent landing rate). The calibration
  table `CAL` was measured greedy on real text with every chain run to depth 8 (the draft head is overconfident:
  p 0.5 lands ~0.41, p 0.85 ~0.71, p 0.99 ~0.95). In-process sweep on real prose (the GLM and infra docs) and code
  (TensorFold modules), 2-6k-token prompts, steady-state decode, interleaved, replies identical across every arm:
  14 rule arms x 3 passes, then the leaders x 4 fresh passes. Winner by the pre-stated rule (no cell worse than -1%,
  mean over the six cells better than +2%): `cum` at T 0.30 with depth 8, tok/s a stream vs production (6 / conf
  0.60), medians: prose C=1/2/4 52.8 / 38.5 / 27.6 -> 54.2 / 39.3 / 27.9, code 75.6 / 48.5 / 34.4 ->
  80.2 / 50.8 / 35.1 (+1.3 to +6.1%, mean +3.1%, every cell's pass range disjoint from production's). The rule was
  chosen and calibrated on those texts, so it was checked again through the HTTP server on `qf_run.py`'s decode
  prompts (`chain_http_ab.sh`: one image, the launcher's env per arm, 3 passes each alternating, cold n-gram tables;
  2026-10-07): per-request tok/s code C=1/2/4 85.3 / 64.4 / 41.1 -> 90.0 / 63.3 / 42.1, prose 74.2 / 50.5 / 32.1 ->
  74.4 / 51.1 / 32.4 (code C=2's aggregate 122.8 -> 123.9); about +1.4% on average there, +5.5% for one stream
  writing code; 8/8 replies byte-identical. Back to v0.6.1's rule: `MTP_DRAFTS=6 TENSORFOLD_CHAIN_RULE=conf`.
- `ple-native/` (2026-10-08): `ssd_table.py` (ple-cache's plus a `load_inline` C++ reader) does every n-gram table
  `pread` in native threads with the GIL released (1 thread per 512 reads, up to 64), then takes the rows in the same
  order: the same files, offsets and bytes. One warm gather of 16,384 ids 232 -> 9.5 ms (a 2,048-row chunk is 32,768); prefill 8k
  / 32k / 128k +47 / +12 / +3% through HTTP, decode unchanged. `TENSORFOLD_PLE_NATIVE=0` restores the Python reads
  (also the fallback, with a line on stderr, when the extension cannot build). `test_ple_native.py` (bit-identical to
  the Python reads on the real 29.8 GiB table) and `mutants_plenative.sh` (5 of 5 killed).
- `mtp-graph/` (2026-10-08): `.../cuda/multi.py` (chain-stop's plus `ChainGraphs`) replays the MTP head's one-row
  chain steps as CUDA graphs, keyed by the stepping streams' MTP cache addresses and a power-of-two context bucket
  (a key is captured on its second sight; an LRU of 32). `TENSORFOLD_MTP_GRAPHS=0` runs them eagerly.
  `test_mtp_graph.py` (3,205 replays bit-identical over poisoned state, contexts crossing 512-token chunks and the
  256-token score grid inside one key) and `mutants_mtpgraph.sh` (7 of 7 killed).
- `ple-defer/` (2026-10-08): `forward.py` (the image's plus deferred staging) and `.../cuda/multi.py` (mtp-graph's
  plus `PLE_DEFER`) read a decode round's n-gram rows when the forward reaches the n-gram layer, behind layers 0-1.
  With `mtp-graph`: decode +1.8 to +2.2% in all six cells (code / prose at 1 / 2 / 4 streams), prefill unchanged.
  `TENSORFOLD_PLE_DEFER=0` stages them first, as v0.6.1. `test_ple_defer.py` and `mutants_pledefer.sh` (3 of 3).
- `hc-split/` (measured, NOT in the image): a bit-exact 4-stream up-mix kernel (production's reduction order
  recovered: 0 of 14,400 sums differ); -8% a call, ~0.1% of a round, below what an HTTP A/B can show.

- `tokenize-nogil/`: `tensorfold/cuda/server.py` from v0.6.1 with every `self.tok.encode(...)` routed through one
  helper, `_encode_ids` (HF tokenizers' `encode_batch_fast` on the one text: the same ids, with Python's GIL
  released and no character offsets), as GLM's patch 0082. `encode` held the GIL for the whole prompt (~0.36 s
  at 256k tokens, glm5.3-flash-dev `scripts/glm53/v18/tok_gil.py`), stopping the engine loop and every other stream at
  each long request's start. `test_tokenize_nogil.py` (in `scripts/glm53/v18/`, run with this tokenizer: ids
  identical over the real code/prose corpus, special tokens, Unicode, both flags; GIL released) passes and
  `mutants_0082.sh` kills its mutants.

- `stock_ab.sh` + `stock_sum.py`: the recipe as shipped (its own image and defaults) against production through the
  live HTTP server, 3 passes an arm alternating, replies byte-identical across arms (2026-10-08).
- `zig_ab.sh` + `zig_sum.py`: the recipe's Zig release (1.0.0, main @ `364f21a` since 2026-10-08: TensorFold's Zig
  engine, the INT4-AutoRound checkpoint, FP8 KV) against production on ONE box (spark4), the same passes; not run (it
  stops production for about an hour). The engines and checkpoints differ, so replies are not compared and both arms'
  speeds come from the client side (time to the first token, then the rest of the reply).
- `zig2_ab.sh <arm>` + `zig_sum.py`: the same comparison on TWO boxes (owner 2026-10-08), run: spark6 serves the Zig
  release (`~/qwen-zig`, its own `start.sh` and defaults) and spark4 keeps serving production; both boxes start
  together, each measuring its own server, and `/health`'s request counter flags outside traffic on production (none
  in any pass; the Zig server's `/health` has no counter, and spark6 has no clients). `gpu_equiv.py` (copy bandwidth,
  bf16 matmul) and `gpu_clocks.sh` (idle CPU, clocks and power under that load) measure how alike the two boxes are:
  copy bandwidth equal (241-245 GB/s), bf16 matmul spark6 ~97-99 against spark4 ~89-92 TFLOPS with neither box busy
  at idle, so compute-bound prefill may favour spark6 by up to ~8%.
- `prefill_fresh.py <arm>`: cold prefill with prompts new to the server (8k / 32k / 128k tokens, 3 each). zig2_ab.sh's
  prefill prompts repeat every pass and the Zig server answers a repeated prompt from its cache (128k in 0.31 s), so
  its prefill rows measure the cache, not prefill.

Result (2026-10-08, thinking off, 512-token replies; decode = the median of 3 passes' medians, per request, client
side; prefill = `prefill_fresh.py`, median of 3):

| | production (spark4) | Zig release (spark6) | Zig / production |
|---|---:|---:|---:|
| code, 1 / 2 / 4 clients | 92.6 / 65.1 / 42.8 tok/s | 105.4 / 77.0 / 55.5 | 1.14 / 1.18 / 1.30 |
| prose, 1 / 2 / 4 clients | 76.1 / 52.1 / 33.0 tok/s | 62.2 / 50.9 / 37.4 | 0.82 / 0.98 / 1.13 |
| prefill 8k / 32k / 128k | 2,723 / 2,693 / 2,311 tok/s | 2,933 / 2,889 / 2,474 | 1.08 / 1.07 / 1.07 |

The Zig release decodes code faster at every concurrency and prose faster at 4 clients, and is 18% slower on prose
at one client. Its prefill lead (7-8%) is no larger than spark6's compute lead (bf16 matmul ~8%), so it shows no
prefill gain of its own. Its pass medians spread more (code at one client 99.4-106.1, production 92.1-92.8). The
checkpoints differ (INT4-AutoRound against MLX 4-bit), so the replies were not compared.

Verify a rebuild before serving it: the golden replies must be byte-identical to the recipe's image (spark4
`~/work/qf_harness.py golden` then `diff`), as they were for the image now serving.
