# Upstream PR (prepared, not submitted): the server tokenizes with Python's GIL released (patch 0018)

Target: `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold`, `main` at `4c0dea8`; no branch, fork or commit made.
`pr.diff` is the whole change (`git apply pr.diff` at `4c0dea8`: 5 files, the patch, `tools/tokenize_check.py`, README
/ CHANGELOG / CREDITS). Placeholder number 0018. Independent of every other PR (`tensorfold/cuda/server.py` only; it
applies alone and under the series, checked). The GLM recipe twin is our prepared `upstream/recipe-pr-tokenize-nogil`
(GLM patch 0087, our 0082).

The same change runs in our build as qwen3.8-flash-next-dev `scripts/qwen38/tokenize-nogil/` (production on spark4 since 2026-10-07
~18:55). With this patch applied, `server.py` equals that layer's file (CRs stripped) except the helper's docstring
tail, which names our repositories; the PR's names the patch and its check instead.

Verified (2026-10-08, spark4, CPU-only containers, production untouched):
- `verify_apply.sh`: applies with the recipe's `patch -p0 --forward` after 0002, alone and under the whole series; the
  result equals our layer file (with the docstring above): exit 0.
- `template_checks.sh` (`git apply`, `bash -n`, `py_compile`, shellcheck adds none): OK.
- `mutants.sh`: patched ALL PASS (9 checks; 873,314 tokens in 1,157 ms with the counting thread at 98% of idle);
  without 0018 it fails (9 checks FAIL); 5 of 5 mutants killed; `exit 0` (log
  `~/QwenPRs/logs/mutants-qwen-pr-tokenize-nogil.log`). Passes with all seven patches applied.
- Leak scan of `pr.diff`: nothing.
- Evidence: `docs/SPEED-CAMPAIGN-2026-10.md` section 2 (the `tokenize-nogil` KEEP item: ~0.36 s at 256k tokens,
  `tok_gil.py`) and spark4 `~/work/nogil-live.log` (golden 6/6, replies equal to the previous image's 14/14).

Awaiting the owner's approval of the text below.

---

**Title:** Patch 0018: no stall at a long request's start: the server tokenizes with Python's GIL released

**Body:**

## Description and Motivation

The server turns each prompt into ids with the tokenizer's `encode` (`App._prepare` for chat and completions, `App.decisions`, the forced think close and the tool-call gate). HF tokenizers' `encode` holds Python's GIL for the whole text: ~0.36 s at 256k tokens on the DGX Spark. The engine loop that schedules every stream's rounds runs in the same process, so while a long prompt is being tokenized, every other stream stops.

This patch routes every one of those calls through one helper, `App._encode_ids(text, special)`, which calls `self.tok.encode_batch_fast([text], add_special_tokens=special)[0].ids`: the same ids, computed with the GIL released, and faster because it skips the character offsets nothing in the server reads.

The ids do not change, so neither do the replies.

`0018` is a placeholder. This PR does not depend on my other PRs, and none of them depends on it.

## Related Issues

None.

---

## Testing

Setup: one DGX Spark (GB10), DGX OS, the recipe at `4c0dea8` with its defaults (`PARALLEL=5`, `CONTEXT=262144`, int8 KV, `PLE_ON_SSD=1`, vision on); on our server it runs with my patches 0012-0015 and `TENSORFOLD_CHAIN_RULE=cum`, `MTP_DRAFTS=8`. The patch was applied with the recipe's own `patch -p0 --forward` onto the recipe image's site-packages (after 0002); I did not run `scripts/prepare.sh` itself.

- `git apply` at `4c0dea8`, `bash -n start.sh stop.sh scripts/*.sh`, `python3 -m py_compile tools/*.py`: pass. shellcheck finds nothing new.
- `tools/tokenize_check.py <tokenizer.json> [dirs]` (CPU, in the image) calls the server's own `App._encode_ids` with the real Qwen3.8 Flash Next tokenizer and compares it with `tok.encode(text, add_special_tokens=flag).ids` for both flags: every .py file of the installed tensorfold package and the recipe checkout's .md / .py / .sh files (~380 files), all of them as one string (3.6 MB), a numbered filler, the 33 added tokens' texts interleaved with words, 200 random Unicode strings (CJK, emoji, combining marks, controls) and edge cases. Qwen's post-processor adds no tokens, so a small template tokenizer checks that `add_special_tokens` passes through. No `self.tok.encode(...)` call may remain in the server. A counting thread must keep at least half its idle rate while ~870k tokens are encoded: it kept 98% (873,314 tokens in 1,157 ms). 9 checks pass with the patch and all 9 fail without it.
- The check fails on each of five mutants of the helper: `encode` instead of the batch call (GIL held), the flag ignored, the flag inverted, the last id dropped, one call site left on `encode`.
- Bits: on our server, six fixed prompts at temperature 0 (seed 1234, thinking off) give byte-identical full replies to the recipe's own image (6/6), and 14 decode replies were identical to the image before this patch.
- In production on our Spark since 2026-10-07.
- Memory: unchanged; the batch call allocates as `encode` does, minus the offsets.

---

## Checklist:

- [x] I have read the README and `scripts/config.sh` and kept my changes consistent with them.
- [x] My pull request has a sound title and description (not something vague like `Update README.md`).
- [x] My change is reproducible and verified (script syntax check, `scripts/prepare.sh`, a launch, or a re-measurement).
- [x] A patch change keeps every reply byte-identical, and I said how I checked it.
- [x] I updated the README and/or `scripts/config.sh` if a setting, default, or measured number changed.
- [x] If my change affects memory, I checked the startup estimate still fits the budget at the default 5 x 262,144 int8 setting.
- [x] Defaults in `scripts/config.sh` still work out of the box; a new setting has a sane fallback like the existing ones.
