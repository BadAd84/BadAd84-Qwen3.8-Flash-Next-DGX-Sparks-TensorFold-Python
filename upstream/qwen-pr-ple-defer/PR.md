# Upstream PR (prepared, not submitted): a decode round's n-gram rows read behind layers 0-1 (patch 0017)

Target: `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold`, `main` at `4c0dea8`; no branch, fork or commit made.
`pr.diff` is the whole change (`git apply pr.diff` at `4c0dea8`: 5 files, the patch, `tools/ple_defer_check.py`, README
/ CHANGELOG / CREDITS). Placeholder number 0017. Depends on 0016 (measured on top of it; its `multi.py` hunks sit on
0012-0016's context), so on 0012, 0013 and 0015 too; not on 0014. Nothing builds on it. This text also carries the
whole series' A/B (the recipe as shipped against all seven patches), since 0017 is the top of the stack.

The same change runs in our build as qwen3.8-flash-next-dev `scripts/qwen38/ple-defer/` (production on spark4 since 2026-10-08
~02:01, image 8e4a7bafb67d). With 0012, 0013, 0015, 0016 and this patch applied, `forward.py` and `multi.py` equal that
layer's files (CRs stripped).

The PR's check is CPU-only (kernels stood in for). The GPU evidence (every staged row set bit-identical, replies
identical with the deferral off) comes from our GPU test `scripts/qwen38/ple-defer/test_ple_defer.py` (serve stopped,
2026-10-08), not part of the PR and not re-run today.

Verified (2026-10-08, spark4, CPU-only containers, production untouched):
- `verify_apply.sh`: applies with the recipe's `patch -p0 --forward` after 0002, 0012, 0013, 0015, 0016 (with and
  without 0014) and under the whole series; the result equals our layer files: exit 0.
- `template_checks.sh` (`git apply`, `bash -n`, `py_compile`, shellcheck adds none): OK.
- `mutants.sh`: patched ALL PASS (4 checks); without 0017 it fails (4 checks FAIL); 6 of 6 mutants killed; `exit 0`
  (log `~/QwenPRs/logs/mutants-qwen-pr-ple-defer.log`). Passes with all seven patches applied.
- Leak scan of `pr.diff`: nothing.
- Evidence: `scripts/qwen38/ple-defer/README.md` (status and the HTTP A/B table); the whole series: `stock_ab.sh` +
  `stock_sum.py` on spark4 today (`~/work/stock-ab.log`, `stockab.jsonl`), as in the task facts.

Awaiting the owner's approval of the text below.

---

**Title:** Patch 0017: a decode round's n-gram rows read while the GPU runs layers 0-1: with 0016, decode +1.8 to +2.2%, same bytes

**Body:**

## Description and Motivation

A decode round's `stage()` looks up every n-gram layer's rows, reads them (from SSD with `PLE_ON_SSD=1`) and copies them to the GPU before the forward starts. The GPU waits for all of it, although the first n-gram layer only comes after layers 0 and 1.

This patch adds `stage(..., defer_ple=True)`. The token ids still go up in `stage`, but the n-gram rows are looked up, read and copied when the forward reaches its first n-gram layer: `ple_block` runs the staging (`b.ple_staging`) before its first kernel, then records the stream's event. So the host reads the rows while the GPU runs the layers before it. The same bytes land in the same buffers ahead of the same kernels on the stream; nothing else changes.

A decode round alone defers. A round that shares its forward with a prompt pass stages the rows first, as before. `TENSORFOLD_PLE_DEFER=0` always stages them first.

The bits do not change, for the reason above.

Measured on one DGX Spark through the server, interleaved A/B, 3 passes an arm, arm order rotated, cold n-gram table every boot, median tok/s a request. "Base" is 0012-0015 and 0018; "0016" adds the MTP graphs; "0016 + 0017" adds this patch:

| Workload | Base | 0016 | 0016 + 0017 |
| --- | ---: | ---: | ---: |
| code, 1 stream | 90.2 | 91.3 | 92.1 (+2.1%) |
| code, 2 streams | 63.6 | 64.5 | 64.9 (+2.0%) |
| code, 4 streams | 42.1 | 42.4 | 42.9 (+1.9%) |
| prose, 1 stream | 74.6 | 75.2 | 76.0 (+1.9%) |
| prose, 2 streams | 51.0 | 51.5 | 51.9 (+1.8%) |
| prose, 4 streams | 32.2 | 32.3 | 32.9 (+2.2%) |
| prefill 8k / 32k / 128k | 2,548 / 2,634 / 2,301 | 2,543 / 2,638 / 2,298 | 2,551 / 2,640 / 2,305 |

Every pass median of the last column is above every pass median of the base in all six decode cells. I did not measure 0017 without 0016.

The whole series (0012-0018) against the recipe as shipped, measured today the same way (the recipe's image and defaults, `MTP_DRAFTS=6` with `conf`, against all seven patches with `TENSORFOLD_CHAIN_RULE=cum` and `MTP_DRAFTS=8`; 3 passes an arm, alternating, cold n-gram table every boot; every reply byte-identical):

| Workload | 1 stream | 2 streams | 4 streams |
| --- | --- | --- | --- |
| code | 79.5 -> 92.7 | 60.0 -> 65.1 | 37.4 -> 43.2 |
| prose | 66.8 -> 76.1 | 46.1 -> 51.9 | 28.5 -> 33.2 |

Prefill at 8k / 32k / 128k: 1,615 / 2,394 / 2,263 -> 2,562 / 2,649 / 2,308 tok/s.

`0017` is a placeholder. This PR depends on #<0016 PR>, and so on #<0012 PR>, #<0013 PR> and #<0015 PR>.

## Related Issues

None.

---

## Testing

Setup: one DGX Spark (GB10), DGX OS, the recipe at `4c0dea8` with its defaults (`PARALLEL=5`, `CONTEXT=262144`, int8 KV, `PLE_ON_SSD=1`, vision on) except my patches as listed in each table, with `TENSORFOLD_CHAIN_RULE=cum` and `MTP_DRAFTS=8` wherever 0015 is in. Patches applied with the recipe's own `patch -p0 --forward` onto the recipe image's site-packages (after 0002); I did not run `scripts/prepare.sh` itself.

- `git apply` at `4c0dea8`, `bash -n start.sh stop.sh scripts/*.sh`, `python3 -m py_compile tools/*.py`: pass. shellcheck finds nothing new.
- Bits: six fixed prompts at temperature 0 (seed 1234, thinking off) give byte-identical full replies to the recipe's own image on every arm (6/6), and the A/B's 8 decode replies (greedy) were identical across arms and passes.
- On the GPU, with the server stopped, my test read every decode round's staged rows back from the GPU buffers and compared them bit for bit with the table's rows for the ids `stage` looked up: every decode round deferred, none lost, 1,535 staged row sets identical. The same requests with the deferral off and on gave identical replies, rounds and accepted drafts. That test needs the GPU and the model, so it is not in this PR.
- `tools/ple_defer_check.py` (CPU, no model) runs the patch's own `stage` and `ple_block` on CPU buffers with a fake n-gram table, the kernels stood in for by recorders that copy the staged buffers at each launch. It checks: with `defer_ple`, `stage` leaves the buffers alone, and `ple_block` stages the rows before its first kernel, the same bytes an immediate staging puts there, and records the event; with no n-gram layer nothing is deferred; the round defers alone, never beside a shared prompt pass, and clears the staging when its forward ends; `TENSORFOLD_PLE_DEFER=0` turns it off. It passes with the patch (4 checks), fails without it (4 of 4), and fails on each of 6 mutants: never deferring, deferring beside a prompt pass, the staging dropped, the staging after the n-gram kernels, the staging not cleared, the deferral off by default.
- In production on our Spark since 2026-10-08.
- Memory: unchanged.

---

## Checklist:

- [x] I have read the README and `scripts/config.sh` and kept my changes consistent with them.
- [x] My pull request has a sound title and description (not something vague like `Update README.md`).
- [x] My change is reproducible and verified (script syntax check, `scripts/prepare.sh`, a launch, or a re-measurement).
- [x] A patch change keeps every reply byte-identical, and I said how I checked it.
- [x] I updated the README and/or `scripts/config.sh` if a setting, default, or measured number changed.
- [x] If my change affects memory, I checked the startup estimate still fits the budget at the default 5 x 262,144 int8 setting.
- [x] Defaults in `scripts/config.sh` still work out of the box; a new setting has a sane fallback like the existing ones.
