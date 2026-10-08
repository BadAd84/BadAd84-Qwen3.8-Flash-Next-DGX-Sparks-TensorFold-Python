# Upstream PR (prepared, not submitted): copies from the context before MTP chains, TensorFold #319 ported to v0.6.1 (patch 0012)

Target: `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold`, `main` at `4c0dea8`; no branch, fork or commit made.
`pr.diff` is the whole change (`git apply pr.diff` at `4c0dea8`: 8 files, the patch, `tools/copy_first_check.py`,
README / CHANGELOG / CREDITS, `scripts/config.sh` and `start.sh` lines for the retired `TENSORFOLD_MTP_COPY`, and the
removal of `tools/test_copy_draft_rows.py`, which tests 0002's copy path that this patch replaces and already fails on
main). Numbered 0012 as a placeholder: open PRs use 0003 (#22, #27, #30) and 0011 (#17); our seven take 0012-0018 so
the series applies in order after 0002. Independent of our other PRs; 0013, 0015, 0016 and 0017 build on it.

The same change runs in our build as qwen3.8-flash-next-dev `scripts/qwen38/pr319/` (production on spark4 since 2026-10-07). With
the patch applied the six files equal that layer's files (CRs stripped), byte for byte.

Verified (2026-10-08, spark4, CPU-only containers, production untouched):
- `verify_apply.sh`: the patch applies with the recipe's `patch -p0 --forward` onto `tensorfold-qwen38:v0.6.1`'s
  site-packages (after 0002), alone and under the whole series, and the result equals our layer files: exit 0.
- `template_checks.sh`: `git apply` at `4c0dea8`, `bash -n`, `py_compile tools/*.py`, shellcheck adds no finding: OK.
- `mutants.sh`: patched ALL PASS (5 checks); without 0012 it fails (5 checks FAIL); 9 of 9 mutants killed; `exit 0`
  (log `~/QwenPRs/logs/mutants-qwen-pr-copy-first.log`). The check also passes with all seven patches applied.
- Leak scan of `pr.diff`: nothing.
- Evidence: speed `docs/SPEED-CAMPAIGN-2026-10.md` section 1 (the #319 row); startup estimate: the recipe
  README's 85.39 GiB, our server's log 85.81 GiB within 115.52 GiB, and the check's geometry 9.356 -> 9.776 GiB.

Awaiting the owner's approval of the text below.

---

**Title:** Patch 0012: copies from the context before MTP chains (TensorFold #319 ported to v0.6.1): a verbatim repeat 153.8 -> ~535 tok/s, same replies

**Body:**

## Description and Motivation

TensorFold #319 by @Arminova made Flash Next on CUDA try a copy from the context before the MTP chain. A reply that repeats its prompt, such as an agent quoting or rewriting a file it just read, otherwise keeps only a few tokens a round. #319 was written for TensorFold 0.6.4 and closed unmerged. This patch ports it to the recipe's v0.6.1, on top of 0002.

What it does, as in #319:

- On one GPU, a round first verifies a copied continuation when the context repeats eight or more tokens (TensorFold's `CopyIndex` from the Qwen3.5 27B engine), else the MTP chain.
- A copy's window starts at 16 rows, doubles while copies land whole (up to 64) and halves after a break (`copies.Copies`, on the 27B engine's `next_copy_rows`).
- With `PARALLEL` >= 2, every stream keeps its chain's rows (depth + 1). Copies share the rest of a round's max(`PARALLEL` x (depth + 1), 64) rows, oldest stream first (`copies.fit`). Every slot's replay rows hold 64, and the admission counts them (`indexed_stream_geometry(rows=, scratch=)`).
- A round's tokens join each stream's context before its next drafts, so `take` runs before `_draft_all`.
- Two ranks keep MTP chains only.

It replaces 0002's copy drafts, the `TENSORFOLD_MTP_COPY` path in `MultiDecoder._draft_all`. So that setting leaves `scripts/config.sh` and the README: copies are on whenever MTP drafts are, on one GPU. `tools/test_copy_draft_rows.py` tested that old path (it already fails on main, with `'SimpleNamespace' object has no attribute '_mtp'`), so this PR removes it. `tools/copy_first_check.py` covers the new path.

Measured on one DGX Spark through the server, against the recipe's own image:

| Reply | Before | After |
| --- | ---: | ---: |
| repeats its prompt word for word | 153.8 tok/s | ~535 tok/s (+248%) |
| a one-word rewrite of its prompt | 141.4 tok/s | ~322 tok/s (+128%) |

Prose, code and prefill stayed within noise. #319's own numbers, on 0.6.4: a file rewrite 3.7x faster (92.4 -> 341.0 tok/s with one stream, 92.9 -> 340.3 with `--parallel 5`).

The bits do not change. Copies are drafts, and every drafted token is checked against the model's own sample, as before.

Memory: the startup estimate at the defaults (5 x 262,144, int8) goes from 85.39 GiB (the README's v0.6.1 figure) to 85.81 GiB, within a 115.52 GiB budget on our Spark. The admission formula's own difference is 0.42 GiB: 9.356 -> 9.776 GiB at 262,144 tokens, from `indexed_stream_geometry` on the checkpoint's config.

`0012` is a placeholder: open PRs use 0003 (#22, #27, #30) and 0011 (#17). Please renumber as you like. This PR does not depend on my other PRs. Four of them (0013, 0015, 0016, 0017) build on it.

## Related Issues

None.

---

## Testing

Setup: one DGX Spark (GB10), DGX OS, the recipe at `4c0dea8` with its defaults (`PARALLEL=5`, `CONTEXT=262144`, int8 KV, `PLE_ON_SSD=1`, vision on, `MTP_DRAFTS=6`, `MTP_CONFIDENCE=0.60`). The patch was applied with the recipe's own `patch -p0 --forward` onto the site-packages of the recipe's image `tensorfold-qwen38:v0.6.1` (the published `v0.6.1-797af1d9df4d`, so after 0002). I did not run `scripts/prepare.sh` itself. Not run at `PARALLEL=1`, where the serial engine now copies too.

- `git apply` at `4c0dea8`, `bash -n start.sh stop.sh scripts/*.sh`, `python3 -m py_compile tools/*.py`: pass. shellcheck finds nothing new.
- Bits: six fixed prompts at temperature 0 (seed 1234, thinking off) give byte-identical full replies with and without the patch (6/6).
- `tools/copy_first_check.py <config.json>` (CPU, no weights, in the image) checks against reference rules written in the check: the copy window (16 rows, doubling to 64, halving after a break, a proposal at most the window less one row); `copies.fit`; `_draft_all` with fake MTP steps (copies drafted, the other streams chained from their own logits rows, each copy asked for after the absorb, with the reply's room); the admission's window and scratch rows; and the wiring (one GPU only, the decoder's rows, take before drafts). It passes with the patch (5 checks), fails without it (5 of 5), and fails on each of 9 mutants: the window never doubling, a proposal filling the whole window, `fit` newest first, `fit` dropping chain rows, logits rows not compacted, the copy's room set to the depth, scratch rows not counted, copies on two ranks, drafts before the take.
- In production on our Spark since 2026-10-07, at the defaults above.
- `tools/test_astra_patches.py` fails the same two checks (`admit take-before-draft`, `native_threads`) with and without this patch.

---

## Checklist:

- [x] I have read the README and `scripts/config.sh` and kept my changes consistent with them.
- [x] My pull request has a sound title and description (not something vague like `Update README.md`).
- [x] My change is reproducible and verified (script syntax check, `scripts/prepare.sh`, a launch, or a re-measurement).
- [x] A patch change keeps every reply byte-identical, and I said how I checked it.
- [x] I updated the README and/or `scripts/config.sh` if a setting, default, or measured number changed.
- [x] If my change affects memory, I checked the startup estimate still fits the budget at the default 5 x 262,144 int8 setting.
- [x] Defaults in `scripts/config.sh` still work out of the box; a new setting has a sane fallback like the existing ones.
