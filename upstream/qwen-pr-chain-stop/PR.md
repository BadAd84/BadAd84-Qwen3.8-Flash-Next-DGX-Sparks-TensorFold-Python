# Upstream PR (prepared, not submitted): where an MTP chain stops (patch 0015)

Target: `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold`, `main` at `4c0dea8`; no branch, fork or commit made.
`pr.diff` is the whole change (`git apply pr.diff` at `4c0dea8`: 5 files, the patch, `tools/chain_stop_check.py`,
README (patch row, a configuration row, a checks row) / CHANGELOG / CREDITS). Placeholder number 0015. Depends on 0012
and 0013 (its `multi.py` hunks sit on theirs); not on 0014 (checked: applies without it). 0016 builds on it.

The same change runs in our build as qwen3.8-flash-next-dev `scripts/qwen38/chain-stop/` (production on spark4 since 2026-10-07
~16:50, with `TENSORFOLD_CHAIN_RULE=cum`, `TENSORFOLD_CHAIN_T=0.30`, `MTP_DRAFTS=8` set by `qwen-start.sh`). With
0012, 0013 and this patch applied, `multi.py` equals that layer's file (CRs stripped) except one comment line, whose
pointer to our README is dropped.

The recipe default stays `conf` (v0.6.1's rule) in this PR; the text offers `cum` as a setting with its numbers. If the
owner prefers, the PR could set `TENSORFOLD_CHAIN_RULE=cum` and `MTP_DRAFTS=8` in `scripts/config.sh` instead (with 0012
in, `MTP_DRAFTS=8` costs no startup memory: the window is 64 rows either way, and our server's log at `MTP_DRAFTS=8`
shows 85.81 GiB).

Verified (2026-10-08, spark4, CPU-only containers, production untouched):
- `verify_apply.sh`: applies with the recipe's `patch -p0 --forward` after 0002, 0012, 0013 (with and without 0014)
  and under the whole series; the result equals our layer file: exit 0.
- `template_checks.sh` (`git apply`, `bash -n`, `py_compile`, shellcheck adds none): OK.
- `mutants.sh`: patched ALL PASS (9 checks); without 0015 it fails (9 checks FAIL); 9 of 9 mutants killed; `exit 0`
  (log `~/QwenPRs/logs/mutants-qwen-pr-chain-stop.log`). Passes with all seven patches applied. The first run had one
  survivor (`cum-ignores-room`): the check's chain driver stopped at the room itself and hid the rule's room test; it
  now runs to the depth and a stream with room for 3 tokens kills it.
- Leak scan of `pr.diff`: nothing.
- Evidence: `docs/SPEED-CAMPAIGN-2026-10.md` section 2 (the `chain-stop` KEEP item and the static depth x
  confidence sweep) and `scripts/qwen38/README.md` (`chain-stop/`).

Awaiting the owner's approval of the text below.

---

**Title:** Patch 0015: TENSORFOLD_CHAIN_RULE, where an MTP chain stops (cum: +5.5% for one stream writing code, +1.4% on average, same replies)

**Body:**

## Description and Motivation

v0.6.1 ends a round's MTP chain at the first draft whose draft-head probability is under `--mtp-confidence`. A static sweep of depth (3-8) and confidence (0.5-0.8) on real prose and code found 6 / 0.60, the recipe's default, the best single setting: prose wants short permissive chains, code long ones. One threshold on one draft's probability cannot serve both, and the draft head is overconfident: a draft it gives 0.5 lands about 41% of the time, 0.85 about 71%, 0.99 about 95%.

This patch adds `TENSORFOLD_CHAIN_RULE`:

- `conf` (the default): v0.6.1's rule, unchanged.
- `cum`: each draft's probability is mapped through a measured landing curve (`CAL`, 14 points; greedy, real prose and code, every chain run to depth 8, 4,935 tested drafts). The chain stops once the chance that every draft so far lands, the product of those calibrated values, falls under `TENSORFOLD_CHAIN_T` (default 0.30). The first draft is always kept, as in v0.6.1.
- `cost`, `adapt`, `acum`: experimental. `cost` keeps a draft while its chance buys more than a verify row's measured cost, and `adapt` / `acum` scale `cost` / `cum` by each stream's recent landing rate. They are not recommended; `cum` won.

Drafts change speed only, so every rule keeps the reply's bits: each drafted token is still checked against the model's own sample.

Measured on one DGX Spark through the server, on prompts the rule was not chosen on, interleaved A/B, 3 passes an arm, cold n-gram tables, median tok/s a request. "Before" is `MTP_DRAFTS=6` with `conf` (the recipe's defaults); "after" is `MTP_DRAFTS=8` with `TENSORFOLD_CHAIN_RULE=cum`:

| Workload | 1 stream | 2 streams | 4 streams |
| --- | ---: | ---: | ---: |
| code, before | 85.3 | 64.4 | 41.1 |
| code, after | 90.0 | 63.3 | 42.1 |
| prose, before | 74.2 | 50.5 | 32.1 |
| prose, after | 74.4 | 51.1 | 32.4 |

About +1.4% on average and +5.5% for one stream writing code. Code at 2 streams lost 1.7% a request while its aggregate rose from 122.8 to 123.9 tok/s. On the texts the rule was chosen and calibrated on (in-process, 14 rule arms x 3 passes, then the leaders x 4 fresh passes), it measured +1.3 to +6.1% in all six cells, mean +3.1%.

This PR leaves the default at `conf`. `TENSORFOLD_CHAIN_RULE=cum MTP_DRAFTS=8 ./start.sh restart` gives the numbers above (`start.sh` passes `TENSORFOLD_*` variables into the container). With 0012 in, `MTP_DRAFTS=8` costs no startup memory: the window is 64 rows either way. I am happy to make `cum` and 8 the defaults if you prefer.

`0015` is a placeholder. This PR depends on #<0012 PR> and #<0013 PR> (its `multi.py` hunks sit on theirs). #<0016 PR> builds on it.

## Related Issues

None.

---

## Testing

Setup: one DGX Spark (GB10), DGX OS, the recipe at `4c0dea8` with its defaults (`PARALLEL=5`, `CONTEXT=262144`, int8 KV, `PLE_ON_SSD=1`, vision on) and my patches 0012 and 0013 in both arms; the arms differ only in `MTP_DRAFTS` and `TENSORFOLD_CHAIN_RULE` as above. Patches applied with the recipe's own `patch -p0 --forward` onto the recipe image's site-packages (after 0002); I did not run `scripts/prepare.sh` itself.

- `git apply` at `4c0dea8`, `bash -n start.sh stop.sh scripts/*.sh`, `python3 -m py_compile tools/*.py`: pass. shellcheck finds nothing new.
- Bits: six fixed prompts at temperature 0 (seed 1234, thinking off) give byte-identical full replies to the recipe's own image with either rule (6/6), and the A/B's 8 decode replies (greedy) were identical across arms.
- `tools/chain_stop_check.py` (CPU, no model) checks the rules against reference implementations written in the check: the settings and their defaults; `conf` decides exactly as v0.6.1 over 20,352 (probability, depth, room, confidence) cases; `_cal` is the `CAL` curve; `cum`, `acum` and `cost` over 7,728 random and edge chains; `_landed` feeds the probe and each stream's trust; `_draft_all` with fake MTP steps chains four streams as `cum` says, one with room for 3 tokens, and starts each round's chains afresh; the round reports drafted and landed rows. It passes with the patch (9 checks), fails without it (9 of 9), and fails on each of 9 mutants: the default rule `cum`, a flat curve, `cum` on raw probabilities, the first draft dropped, `cum` past the room, chains not reset, trust frozen, landed rows off by one, `cost` keeping every draft.
- In production on our Spark since 2026-10-07, with `TENSORFOLD_CHAIN_RULE=cum` and `MTP_DRAFTS=8`.
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
