# Upstream PR (prepared, not submitted): n-gram reads from the page cache, and read hints (patch 0013)

Target: `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold`, `main` at `4c0dea8`; no branch, fork or commit made.
`pr.diff` is the whole change (`git apply pr.diff` at `4c0dea8`: 5 files, the patch, `tools/ple_cache_check.py`, README
/ CHANGELOG / CREDITS). Placeholder number 0013. Depends on 0012 (copy-first): its `multi.py` hunks sit in 0012's
`_draft_all`, and the hints cover 0012's copies. 0014 and 0015 build on it.

The same change runs in our build as qwen3.8-flash-next-dev `scripts/qwen38/ple-cache/` (production on spark4 since 2026-10-07
~12:40). With 0012 and this patch applied, `multi.py` and `ssd_table.py` equal that layer's files (CRs stripped).

Verified (2026-10-08, spark4, CPU-only containers, production untouched):
- `verify_apply.sh`: applies with the recipe's `patch -p0 --forward` after 0002 and 0012, alone and under the whole
  series; the result equals our layer files: exit 0.
- `template_checks.sh` (`git apply`, `bash -n`, `py_compile`, shellcheck adds none): OK.
- `mutants.sh`: patched ALL PASS (7 checks); without 0013 it fails (6 checks FAIL); 8 of 8 mutants killed; `exit 0`
  (log `~/QwenPRs/logs/mutants-qwen-pr-ple-page-cache.log`). Passes with all seven patches applied.
- Leak scan of `pr.diff`: nothing.
- Evidence: `docs/SPEED-CAMPAIGN-2026-10.md` section 2 (the `ple-cache` KEEP item and its profile bullet) and
  `scripts/qwen38/README.md` (`ple-cache/`).

Awaiting the owner's approval of the text below.

---

**Title:** Patch 0013: decode +9 to +12% with the n-gram tables on SSD: cached reads on the engine thread, the next round's rows hinted to the page cache

**Body:**

## Description and Motivation

With `PLE_ON_SSD=1`, every decode round's `stage()` reads its n-gram rows from the checkpoint files through `SSDTable.gather`. Every read went through a pool of 16 Python threads. An in-process profile of steady decode on prose showed these gathers leaving the GPU idle for 3.6 / 8.0 / 10.0% of decode time at 1 / 2 / 4 streams. The cause was the threads, not the disk. Each read cost about 9 us of handing work between threads under the GIL, against about 0.4 us for the read itself when its page is cached, so a decode window's gather took ~1.3 ms even with every page cached.

This patch changes two things:

- `SSDTable.gather` does a decode-sized gather's reads (up to 4,096) on the calling thread first, with `RWF_NOWAIT`. A read the page cache holds completes there; a read that would wait on the disk, or the rest of a partly cached one, goes to the threads as before. Larger gathers (prompt pieces) go to the threads as before.
- `MultiDecoder` hints the next round's rows with `POSIX_FADV_WILLNEED`, on one helper thread, while the GPU runs the MTP steps: each stream's first token after the accept, each copy (0012), then each draft as the chain picks it. `SSDTable.will_need` advises exactly each row's three parts.

The bytes do not change. Both paths read the same file bytes; a hint only starts a page read early.

`TENSORFOLD_PLE_NOWAIT=0` sends every read to the threads again, and `TENSORFOLD_PLE_HINTS=0` turns the hints off. `start.sh` already passes `TENSORFOLD_*` variables into the container. With the tables in memory (`PLE_ON_SSD=0`) nothing changes.

Measured on one DGX Spark through the server, interleaved A/B, 3 passes an arm, median tok/s a request (the pass spreads did not overlap):

| Workload | 1 stream | 2 streams | 4 streams |
| --- | ---: | ---: | ---: |
| prose, before | 66.2 | 45.6 | 28.4 |
| prose, after | 73.9 | 50.1 | 31.9 |
| code, before | 77.7 | 58.8 | 36.9 |
| code, after | 85.1 | 64.0 | 41.0 |

That is +9 to +12% in every cell. Every reply in the A/B was byte-identical.

`0013` is a placeholder. This PR depends on #<0012 PR> (copy-first): its `multi.py` hunks sit in 0012's `_draft_all`. Patches 0014 and 0015 build on this one.

## Related Issues

None.

---

## Testing

Setup: one DGX Spark (GB10), DGX OS, the recipe at `4c0dea8` with its defaults (`PARALLEL=5`, `CONTEXT=262144`, int8 KV, `PLE_ON_SSD=1`, vision on, `MTP_DRAFTS=6`, `MTP_CONFIDENCE=0.60`), with 0012 applied in both arms. Patches applied with the recipe's own `patch -p0 --forward` onto the recipe image's site-packages (after 0002); I did not run `scripts/prepare.sh` itself.

- `git apply` at `4c0dea8`, `bash -n start.sh stop.sh scripts/*.sh`, `python3 -m py_compile tools/*.py`: pass. shellcheck finds nothing new.
- Bits: six fixed prompts at temperature 0 (seed 1234, thinking off) give byte-identical full replies to the recipe's own image (6/6), and every reply of the A/B above (greedy) was identical across arms.
- `tools/ple_cache_check.py [dir]` (CPU, no model) writes a small n-gram table (three safetensors files, six shards, ~25 MB) to a directory on a real disk and checks: gathers return exactly the memory-mapped `HostTable`'s rows, cold and warm, for decode, prompt, adjacent-run, shard-boundary and repeated id sets; a warm decode gather sends no read to the threads, a cold one hands its misses over; a partly cached read keeps its cached bytes and hands on the rest; `will_need` advises each row's three parts and nothing outside the table; `_draft_all`, with fake MTP steps, hints exactly the rows the next round's `stage` looks up; both switches work. It passes with the patch (7 checks), fails without it (6 of 7), and fails on each of 8 mutants: cached reads off, a partly read rest misplaced, misses dropped, `TENSORFOLD_PLE_NOWAIT` default off, advice at the wrong offset, only the last row hinted, the chain steps' hints missing, `TENSORFOLD_PLE_HINTS` default off.
- In production on our Spark since 2026-10-07.
- Memory: unchanged (one helper thread; no buffers).

---

## Checklist:

- [x] I have read the README and `scripts/config.sh` and kept my changes consistent with them.
- [x] My pull request has a sound title and description (not something vague like `Update README.md`).
- [x] My change is reproducible and verified (script syntax check, `scripts/prepare.sh`, a launch, or a re-measurement).
- [x] A patch change keeps every reply byte-identical, and I said how I checked it.
- [x] I updated the README and/or `scripts/config.sh` if a setting, default, or measured number changed.
- [x] If my change affects memory, I checked the startup estimate still fits the budget at the default 5 x 262,144 int8 setting.
- [x] Defaults in `scripts/config.sh` still work out of the box; a new setting has a sane fallback like the existing ones.
