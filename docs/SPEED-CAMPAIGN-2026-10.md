# Speed campaign, October 2026 - where the Qwen3.8 Flash Next serve spends its time

Code-level profiling of the Qwen serve on spark4 (2026-10-06/08, owner-directed). Every speed claim below is
MEASURED and A/B'd with byte-identical replies (golden 6/6), or labelled as an estimate. The live config and
each layer's evidence are in `scripts/qwen38/README.md`. The GLM part of this campaign, and the profiling
instruments that worked or did not (engine profiles, py-spy, nsys on these boxes), are in glm5.3-flash-dev
`docs/GLM53-SPEED-CAMPAIGN-2026-10.md` (split 2026-10-08, when each serve got its own repo).

## 1. Results that changed production

| Serve | Change | Measured | Status |
|---|---|---|---|
| Qwen (spark4) | TensorFold PR #319 ported to the recipe's v0.6.1: copy drafts before the MTP chain, windows growing to 64 rows | verbatim repeat 153.8 -> ~535 tok/s (+248%), a one-word rewrite 141.4 -> ~322 (+128%); prose / code / prefill within noise; golden 6/6 byte-identical | production (in `tensorfold-qwen38:v061-ots`) |
| Qwen | `ple-cache`, `chain-stop`, `tokenize-nogil` (section 3) | per request +9 to +12% (ple-cache), code at one stream +5.5% / ~+1.4% on average (chain-stop), no stall at a long request's start (nogil); golden 6/6 | production (`tensorfold-qwen38:v061-ots`) |
| Qwen | `ple-native` (the n-gram table reads in native threads, GIL released) | prefill 8k / 32k / 128k +47 / +12 / +3%; decode unchanged; golden 6/6 | production (2026-10-08 ~00:12) |
| Qwen | `mtp-graph` (the MTP chain steps as CUDA graphs) + `ple-defer` (the n-gram rows read behind layers 0-1) | decode +1.8 to +2.2% in all six cells (code / prose at 1 / 2 / 4 streams); prefill unchanged; golden 6/6 | production (2026-10-08 ~02:01, image 8e4a7bafb67d) |

## 2. Qwen3.8 Flash Next (spark4, single Spark)

Measured on the recipe defaults (5 streams, int8 KV, MTP 6/0.60), greedy, thinking off, median of 3: decode prose
64.6 / 40.2 / 28.1 tok/s a request at 1 / 2 / 4 concurrent, code 83.2 / 62.1 / 39.7; prefill 1,985 / 2,565 / 2,279
tok/s at 8k / 32k / 128k (the recipe's published figures within +-15%, different prompts).

- **Profile:** a round costs ~30 ms fixed plus ~3-4 ms a verified row. In-process attribution (2026-10-07, prose,
  steady decode): GPU busy 85.1 / 84.5 / 84.2% at 1 / 2 / 4 streams; the largest GPU-idle cause was `stage()`'s
  n-gram (PLE) table reads from SSD, 3.6 / 8.0 / 10.0% of decode wall, and the cause was thread overhead, not the
  disk: every gather went through a 16-thread pool, ~9 us of GIL handoff a read against 0.4 us for a cached read.
  Kernel launch gaps were 5.8 / 2.9 / 1.6% of wall, so a graphed multi-stream round recovers at most ~2.5 ms a
  round at 1 stream (its capture keys explode with PR #319's 1-64-row copy windows): NOT the largest lever.
- **KEEP: `ple-cache` (`scripts/qwen38/ple-cache/`, in production since 2026-10-07 ~12:40):** a decode gather's
  cached reads happen on the calling thread with `RWF_NOWAIT` (misses still go to the pool) and the next round's
  rows are hinted with `POSIX_FADV_WILLNEED` while the GPU drafts. Interleaved HTTP A/B, 3 passes an arm: prose
  66.2 / 45.6 / 28.4 -> 73.9 / 50.1 / 31.9 tok/s a request at 1 / 2 / 4 streams, code 77.7 / 58.8 / 36.9 -> 85.1 /
  64.0 / 41.0 (+9 to +12%, pass spreads disjoint); golden 6/6 and every A/B reply byte-identical (golden re-run
  independently on the deployed serve: 6/6; a spot-check pass 86.2 / 64.6 / 41.3 code, 75.1 / 50.9 / 32.3 prose).
  Off switches: `TF_PLE_NOWAIT=0`, `TF_PLE_HINTS=0`.
- **The dense 4-bit `qmm_kernel` is at the memory roofline** (F58): 12-13 ms a round, ~30% of device time, but
  94-106% of spark4's measured 240 GB/s at every served shape and 1-32 rows: DROP (no kernel headroom).
- **Static MTP depth x confidence (2026-10-07): no change.** 20 arms (depth 3-8 x confidence 0.5-0.8) x 3 passes on
  real prose and code: prose wants shallow permissive chains (+2-3%), code deep ones; 6 / 0.60 is the best single
  setting. An oracle that never drafts a rejected token would save 9-16% of a round (the ceiling).
- **KEEP: `chain-stop` (`scripts/qwen38/chain-stop/`, in production since 2026-10-07 ~16:50):** a round's MTP chain
  stops once the calibrated chance that all its drafts land falls under 0.30 (`TENSORFOLD_CHAIN_RULE=cum`,
  `TENSORFOLD_CHAIN_T=0.30`, depth 8; the draft head is overconfident, so its probabilities are mapped through a
  measured landing curve). In-process on the texts it was chosen on: +1.3 to +6.1% in all six cells (mean +3.1%).
  Through HTTP on other prompts (`chain_http_ab.sh`): code C=1/2/4 85.3 / 64.4 / 41.1 -> 90.0 / 63.3 / 42.1, prose
  74.2 / 50.5 / 32.1 -> 74.4 / 51.1 / 32.4 tok/s a request (about +1.4% on average, +5.5% one stream writing code;
  code C=2's aggregate +0.9%). Replies byte-identical everywhere, golden 6/6.
- **KEEP: `tokenize-nogil` (`scripts/qwen38/tokenize-nogil/`, 2026-10-07 ~18:55), as GLM's patch 0082:** the
  server's `encode` held Python's GIL for the whole prompt (~0.36 s at 256k tokens, `tok_gil.py`), stopping every
  other stream at a long request's start; `encode_batch_fast` on the one text gives the same ids with the GIL
  released and ~30% faster. Test and mutants as GLM's, with Qwen's tokenizer; golden 6/6.
- **KEEP: `ple-native` (`scripts/qwen38/ple-native/`, production since 2026-10-08 ~00:12):** the n-gram table reads
  in native threads with the GIL released (a `load_inline` C++ reader; the same files, offsets and bytes). Measured
  cause: a 2,048-row prompt chunk's gather (32,768 ids, ~49k preads through 16 Python threads) took ~230 ms warm and
  left the GPU idle 31% of an 8k prefill. One warm gather of 16,384 ids 232 -> 9.5 ms; interleaved HTTP A/B (3 passes, cold table every
  boot): prefill 8k / 32k / 128k 1,735 / 2,354 / 2,230 -> 2,547 / 2,637 / 2,297 tok/s, decode within +-1%; golden
  6/6, test 14/14 bit-identical on the real table, mutants 5/5.
- **KEEP: `mtp-graph` + `ple-defer` (production since 2026-10-08 ~02:01, image 8e4a7bafb67d):** the MTP chain steps
  replayed as CUDA graphs (keyed by cache addresses and a power-of-two context bucket), and a decode round's n-gram
  rows read behind layers 0-1. Interleaved HTTP A/B, 3 passes: code C=1/2/4 90.2 / 63.6 / 42.1 -> 92.1 / 64.9 /
  42.9, prose 74.6 / 51.0 / 32.2 -> 76.0 / 51.9 / 32.9 tok/s a request (+1.8 to +2.2%, every pass median above
  every production one), prefill unchanged, golden 6/6, 8/8 replies identical; mutants 7/7 and 3/3.
- **Measured negatives (2026-10-08):** the expert kernels run at 1.00-1.03x a read-only kernel over the same
  experts' bytes (real routing replayed; 42% of a 1-stream round, 60% at 4): no headroom, only fewer distinct experts
  a round would cut them. A first-draft floor under `cum` 0.30 (0.10 / 0.15 / 0.20): +0.5 / +0.1 / +0.4%, zero-draft
  rounds are 0-6%. Upload formats: 12-42 us a round. Triton knob tuning of the HC read-out kernels keeps bits only at
  `num_warps` 4 (-5% at best); a bit-exact 4-stream up-mix (`hc-split/`, production's reduction order recovered)
  -8% a call, ~0.1% of a round: not shipped. Vectorizing `gdn_multi.Tables`' pointer loop: <= 0.25% of a round.
- **Next candidates:** GPU idle is 8.4% at one stream (~4.4 ms a round): verify-forward launch gaps (~1.2 ms a
  round), the picks read-back before each MTP step (~1.1 ms), the sampling tail (~1.2 ms) - each needs a graphed
  verify forward or GPU-side draft chaining.
- **PR verdicts** (each ported to v0.6.1 as an image FROM the recipe's, golden byte-check, then measured): #319 KEPT
  (above); #413 (batched n-gram gathers) no reliable gain in its A/B (that A/B's spread was large; the in-process
  profile later showed the gathers do NOT overlap the GPU - `ple-cache` above fixed them a different way); #422
  (per-capture `gc.collect`) and #250 (`--mtp-cost`) are `--parallel 1` features the 5-stream config never runs;
  #387's headline fix is unreachable on this checkpoint (`converges()` true, decode share 0).
- Decode speed varies run to run by up to ~25% on this box (prose 51-68 tok/s across identical runs), so a single A/B
  under ~10% is noise unless it repeats across arms.
