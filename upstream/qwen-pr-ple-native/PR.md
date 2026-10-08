# Upstream PR (prepared, not submitted): the n-gram reads in native threads (patch 0014)

Target: `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold`, `main` at `4c0dea8`; no branch, fork or commit made.
`pr.diff` is the whole change (`git apply pr.diff` at `4c0dea8`: 5 files, the patch, `tools/ple_native_check.py`, README
/ CHANGELOG / CREDITS; CREDITS' PyTorch line already named "the native n-gram reader in patch 0003", which no patch on
main holds; it now names 0014). Placeholder number 0014. Depends on 0013 (its `ssd_table.py` hunks sit on 0013's), so
on 0012 too. Nothing builds on it: 0015-0017 apply with or without it (checked).

The same change runs in our build as qwen3.8-flash-next-dev `scripts/qwen38/ple-native/` (production on spark4 since 2026-10-08
~00:12). With 0012-0014 applied, `ssd_table.py` equals that layer's file (CRs stripped) except the extension's build
name: ours is `tensorfold_ple_gather_ots`, the PR's `tensorfold_ple_gather`.

Verified (2026-10-08, spark4, CPU-only containers, production untouched):
- `verify_apply.sh`: applies with the recipe's `patch -p0 --forward` after 0002, 0012 and 0013, alone and under the
  whole series; the result equals our layer file (with the name above): exit 0. The series without 0014 applies too.
- `template_checks.sh` (`git apply`, `bash -n`, `py_compile`, shellcheck adds none): OK.
- `mutants.sh`: patched ALL PASS (4 checks); without 0014 it fails (2 checks FAIL); 6 of 6 mutants killed; `exit 0`
  (log `~/QwenPRs/logs/mutants-qwen-pr-ple-native.log`). Passes with all seven patches applied. One check run,
  including building the reader, took ~12 s.
- Leak scan of `pr.diff`: nothing.
- Evidence: `scripts/qwen38/ple-native/README.md` (the measured bullets) and `docs/SPEED-CAMPAIGN-2026-10.md`
  section 2 (the `ple-native` KEEP item). The base of its A/B was our image of that night (0012, 0013, 0015, 0018 with
  `TENSORFOLD_CHAIN_RULE=cum`, `MTP_DRAFTS=8`).

Awaiting the owner's approval of the text below.

---

**Title:** Patch 0014: prefill with the n-gram tables on SSD +47% at 8k: the table reads in native threads, the same bytes

**Body:**

## Description and Motivation

With `PLE_ON_SSD=1`, every n-gram lookup reads its rows from the checkpoint files: one `pread` per run of adjacent rows, for the words, scales and biases. After 0013 a decode window's cached reads run on the engine thread, but a prompt piece's reads still go through 16 Python threads, and each read still costs Python work under the GIL. A 2,048-row piece holds 32,768 n-gram ids and needs tens of thousands of reads. In-process, an 8k-token prompt left the GPU idle for 31% of its prefill while it waited for these reads.

This patch builds the same reads as a table (file, offset, length, place in one output buffer) and hands it to a small C++ extension, built once at startup with `torch.utils.cpp_extension.load_inline` into the kernel cache. The extension does every `pread` in native threads with the GIL released: one thread up to 1,023 reads (a decode window), then one per 512 reads, up to 64 (a cold prompt piece waits on the disk, so more reads in flight). The rows are then taken in the same order as before.

The bytes do not change: the same files, offsets and lengths, read into the same rows. `TENSORFOLD_PLE_NATIVE=0` restores the Python reads. They are also the fallback, with one line on stderr, if the extension cannot build.

Measured on one DGX Spark:

- One gather of 16,384 ids, warm pages, best of 5: 232 -> 9.5 ms.
- Through the server, interleaved A/B, 3 passes an arm, every boot from a cold n-gram table, median tok/s (range):

| Prompt | Before | After |
| ---: | ---: | ---: |
| 8k | 1,735 (1,626-1,907) | 2,547 (2,545-2,549) |
| 32k | 2,354 (2,354-2,449) | 2,637 (2,632-2,640) |
| 128k | 2,230 (2,229-2,246) | 2,297 (2,293-2,301) |

Decode stayed within 1% (code at 1 / 2 / 4 streams 90.6 / 63.4 / 41.8 -> 89.9 / 63.3 / 42.1 tok/s a request, prose 74.4 / 51.0 / 32.3 -> 74.1 / 50.8 / 32.3).

This also makes CREDITS' PyTorch line true again: it named "the native n-gram reader in patch 0003", which no patch on main holds.

`0014` is a placeholder. This PR depends on #<0013 PR> (page cache), whose `ssd_table.py` it builds on, and so on #<0012 PR>. None of my other PRs depends on this one.

## Related Issues

None.

---

## Testing

Setup: one DGX Spark (GB10), DGX OS, the recipe at `4c0dea8` with its defaults (`PARALLEL=5`, `CONTEXT=262144`, int8 KV, `PLE_ON_SSD=1`, vision on) except, in both arms of the A/B, my patches 0012, 0013, 0015 and 0018 with `TENSORFOLD_CHAIN_RULE=cum` and `MTP_DRAFTS=8`. Patches applied with the recipe's own `patch -p0 --forward` onto the recipe image's site-packages (after 0002); I did not run `scripts/prepare.sh` itself.

- `git apply` at `4c0dea8`, `bash -n start.sh stop.sh scripts/*.sh`, `python3 -m py_compile tools/*.py`: pass. shellcheck finds nothing new.
- Bits: six fixed prompts at temperature 0 (seed 1234, thinking off) give byte-identical full replies to the recipe's own image (6/6), and the A/B's 8 decode replies (greedy) were identical across arms.
- `tools/ple_native_check.py [dir]` (CPU, no model; builds the reader) writes a small n-gram table (three safetensors files, six shards, ~25 MB) to a directory on a real disk and checks: the reader builds and every gather uses it; `TENSORFOLD_PLE_NATIVE=0` turns it off; its read table equals the Python path's reads; every gather returns exactly the memory-mapped `HostTable`'s rows and the Python reads', cold and warm, for decode, prompt, long-run (past the 1 MiB read limit), shard-boundary and repeated id sets; a table file cut short raises. It passes with the patch (4 checks), fails without it (2 of 4), and fails on each of 6 mutants: the parts written at offset 0, one row a run, the parts overlapping, every other read skipped, the reader off by default, the reader built but not used.
- On the checkpoint's real 29.8 GiB table, my earlier test compared the native reads with the Python reads bit for bit over 7 id sets, cold and warm: 14/14.
- In production on our Spark since 2026-10-08.
- Memory: unchanged (a per-gather buffer the size of the rows read, as before).

---

## Checklist:

- [x] I have read the README and `scripts/config.sh` and kept my changes consistent with them.
- [x] My pull request has a sound title and description (not something vague like `Update README.md`).
- [x] My change is reproducible and verified (script syntax check, `scripts/prepare.sh`, a launch, or a re-measurement).
- [x] A patch change keeps every reply byte-identical, and I said how I checked it.
- [x] I updated the README and/or `scripts/config.sh` if a setting, default, or measured number changed.
- [x] If my change affects memory, I checked the startup estimate still fits the budget at the default 5 x 262,144 int8 setting.
- [x] Defaults in `scripts/config.sh` still work out of the box; a new setting has a sane fallback like the existing ones.
