# ple-native - the n-gram (PLE) table reads in native threads

`tensorfold/families/qwen4_exp/ssd_table.py` = ple-cache's `ssd_table.py` plus a native reader. With
`--ple-on-ssd`, every n-gram lookup reads its table rows from the checkpoint's files (`SSDTable.gather`: one `pread`
per run of adjacent rows, for the words, scales and biases parts). ple-cache did a decode window's cached reads on the
engine thread (`RWF_NOWAIT`) and every other read through 16 Python threads; each read still cost Python work under
the GIL. A prompt chunk (2,048 rows x 16 heads = 32,768 ids, ~49k reads) therefore took ~230 ms warm and ~520 ms
cold, and an 8k-token prompt waited ~1.3 s of its ~4.3 s on these reads (in-process, 2026-10-07: GPU idle 31% of the
8k prefill, two gaps of ~640 ms between host-to-device copies).

The change: `gather` builds the same reads as an int64 table (fd, file offset, bytes, offset in one output buffer
holding the three parts) and hands it to a small C++ extension (`load_inline`, built once at table open into
`TORCH_EXTENSIONS_DIR` as `tensorfold_ple_gather_ots`) that does every `pread` in native threads with the GIL
released: 1 thread up to 1,023 reads (a decode window), one per 512 reads up to 64 (a cold prompt chunk waits on
the disk, so more reads in flight). Same files, same offsets, same bytes; the rows are then taken in the same order
(`out[inverse]`). `TENSORFOLD_PLE_NATIVE=0` restores ple-cache's Python reads; a build failure falls back to them with a
line on stderr.

Measured (spark4, 2026-10-07):
- one gather, warm pages, best of 5: 16,384 ids 232 ms -> 9.5 ms; 490 ids (a 4-stream window) 0.91 -> 0.48 ms;
  112 ids (1 stream) 0.23 -> 0.20 ms; 8 ids equal (`probe_native.py`, lane scratch).
- prefill in-process (`prefill_ple.py`, random-word prompts, 2 runs an arm interleaved, cold = the table's pages
  dropped first): 8k cold 4.36-4.79 s -> 3.17-3.36 s (1,674-1,837 -> 2,385-2,525 tok/s), 8k warm 3.87-4.30 ->
  3.03-3.05 s; 32k cold 13.09-13.84 -> 12.31-12.41 s, 32k warm 12.76-13.07 -> 12.02-12.05 s. Gather seconds summed
  over the prompt: 8k 4.2-4.6 -> 0.9-1.3, 32k 10.8-11.0 -> 2.5-2.6 (the read-ahead thread hid most of them at 32k).
- through the HTTP server (`http_ab.sh`, production's image vs this build, 3 passes alternating, cold n-gram table
  every boot, qf_run.py prompts): prefill 8k 1,735 (1,626-1,907) -> 2,547 (2,545-2,549) tok/s, 32k 2,354
  (2,354-2,449) -> 2,637 (2,632-2,640), 128k 2,230 (2,229-2,246) -> 2,297 (2,293-2,301); decode unchanged within
  +-1% (code C=1/2/4 90.6 / 63.4 / 41.8 -> 89.9 / 63.3 / 42.1, prose 74.4 / 51.0 / 32.3 -> 74.1 / 50.8 / 32.3);
  golden 6/6 on both arms, 8/8 decode replies byte-identical. In production since 2026-10-08 ~00:12 (image
  422254c9ef0b; rollback `tensorfold-qwen38:v061-ots-prenative` = 55a137f8a0e8); golden 6/6 on the deployed serve.
- `test_ple_native.py`: 14/14 PASS; `mutants_plenative.sh`: 5/5 killed, exit 0.

Files:
- `test_ple_native.py` (CPU, the checkpoint's real table): every gather through the native reader against
  ple-cache's Python reads, bit for bit, over decode-sized, prompt-chunk, adjacent-run, shard-boundary and
  unsorted-with-repeats id sets, cold and warm; FAILs if the native reader is not built or not engaged.
- `mutants_plenative.sh <ssd_table.py> <test_ple_native.py> [image]`: 5 mutants (parts written at offset 0, one row
  a run, parts overlapping, every other read skipped, the reader off), each built under its own extension name.
  Not mutated, with the measured reason: the short-read loop (`pread` of a regular file returns the full count short
  of end of file, so no test can make it fail) and the thread-count policy (speed only).

TensorFold's Apache-2.0 license applies as for `pr319/`.
