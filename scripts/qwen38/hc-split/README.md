# hc-split - the hyper-connection up mix with the four streams in one dot (NOT in production)

**Status (2026-10-08): bit-exact, measured, not shipped: the bit-exact kernel saves 1.4 us of an 18 us call, about
0.08 ms of a ~59 ms 1-stream round (~0.1%), far under what an HTTP A/B resolves.**

`tensorfold/families/qwen4_exp/cuda/qmm.py` = the image's `qmm.py` plus `_qmm_upmix4`, launched by `hc_upmix` for 4
streams (`TENSORFOLD_HC_WIDE=0` launches v0.6.1's `_qmm_upmix`). v0.6.1's kernel gives each (row block, 32 dims)
program all four streams in turn: 80 programs on 48 SMs, each a serial chain of 40 [16, 32] x [32, 32] dots (K 320 =
10 groups a stream), 18.1-18.5 us a call against ~8.5 us for its ~2 MB of weights at the memory roofline.
`_qmm_upmix4` loads the program's 4 x 32 weight rows as one [128, 32] operand (column n = dim n // 4 of stream n % 4)
and runs one [16, 32] x [32, 128] dot a group (a chain of 10), then takes the four stream products apart with
`tl.split` and adds them in stream order from zero.

Bits, measured (`test_hc_split.py`, `bench_upmix_*.py` in the lane scratch):
- The mixed values are bit-identical in every launch setting tried (warps 2/4/8, stages 2-4).
- The 32-dim group sums were NOT, at large inputs (3-18 of 1,280 sums differ at activation scale 10-16): `tl.sum`
  reduces in the reduced tensor's layout's order, and the new kernel's layout differs. Production's order was then
  recovered exactly by searching every Triton-reduce order (which index bits a thread sums left to right, then the
  pairwise order of the remaining bits) against production's sums: 0 of 14,400 differ for "each run of 4 dims left
  to right, then the 8 runs pairwise, run bit 2 first, then bit 1, then bit 0"; the next-best order differs in 10.
  `_qmm_upmix4` writes the sums in that order (size-2 `tl.sum`s are single adds, so the pairwise stages are exact).
- `test_hc_split.py` (synthetic: 96 launches, rows 1-16, 6 input sets incl. zeros and negative zeros, a failure ends
  the process before the engine is built; then every eager up mix of the serve run both ways on real weights and
  activations: 98,650 calls over test_mtp_graph.py's requests, rows 1-16; then replies, rounds and accepted drafts
  identical with v0.6.1's kernel) on the final file, in production's image (mtp-graph + ple-defer): exit 0, 18 PASS.
  `mutants_hcsplit.sh <this dir> ../mtp-graph/test_mtp_graph.py tensorfold-qwen38:v061-ots`: exit 0, 7 of 7 killed
  (`sum-as-tree`, `sum-order`, `sum-from-v0`, `stream-major-columns`, `wide-off`, `group-sum-tl`,
  `group-run-tree`), none hung. Not mutated: `WIDE_WARPS` / `WIDE_STAGES` (every setting tried keeps the bits).

Speed (quiet GPU, graph of 48 launches over 48 weight copies, median of 5 interleaved rounds, us a call):

| rows | v0.6.1 | four-stream dot, tl.sum (bits differ) | four-stream dot, exact order (permutes) | exact order (size-2 sums) |
|---|---|---|---|---|
| 1 | 18.1-18.4 | 15.4 (w8) | 17.45 | 16.63 |
| 4 | 18.2-18.4 | 15.4 | 17.47 | 16.81 |
| 9 | 18.2-18.5 | 15.5 | 17.50 | 16.88 |
| 16 | 18.2-18.5 | 16.5 | 17.60 | 16.77 |

Also measured, bit-identical and not shipped: the up mix split by stream (`bench_upmix_split.py`: 320 programs plus
a mix kernel, 13.9-15.0 us, one more launch a call) and the same split in ONE launch with a per-(row block, dims)
semaphore whose last-arriving program mixes (13.4-14.0 us, -25%): its gain, ~4.6 us x ~58 calls = ~0.27 ms a round
(~0.45%), rests on cross-program memory ordering (acq_rel atomics, .cg reloads) that no test can make fail
deterministically, for a gain an HTTP A/B cannot resolve.

TensorFold's Apache-2.0 license applies as for `pr319/`.
