# Upstream PR (prepared, not submitted): the MTP head's chain steps as CUDA graphs (patch 0016)

Target: `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold`, `main` at `4c0dea8`; no branch, fork or commit made.
`pr.diff` is the whole change (`git apply pr.diff` at `4c0dea8`: 5 files, the patch, `tools/mtp_graph_check.py`, README
/ CHANGELOG / CREDITS). Placeholder number 0016. Depends on 0015 (its `multi.py` hunks sit on 0015's), so on 0012 and
0013 too; not on 0014. 0017 builds on it.

The same change runs in our build as qwen3.8-flash-next-dev `scripts/qwen38/mtp-graph/` (production on spark4 since 2026-10-08
~02:01, image 8e4a7bafb67d, with `TENSORFOLD_CHAIN_RULE=cum`, `MTP_DRAFTS=8`). With 0012, 0013, 0015 and this patch
applied, `multi.py` equals that layer's file (CRs stripped).

The PR's check is CPU-only (CUDA graphs stood in for); the bit-identity of a replay on the GPU comes from our GPU test
`scripts/qwen38/mtp-graph/test_mtp_graph.py` (run with the serve stopped, 2026-10-08), which is not part of the PR. It
was not re-run today: spark4 serves production.

Verified (2026-10-08, spark4, CPU-only containers, production untouched):
- `verify_apply.sh`: applies with the recipe's `patch -p0 --forward` after 0002, 0012, 0013, 0015 (with and without
  0014) and under the whole series; the result equals our layer file: exit 0.
- `template_checks.sh` (`git apply`, `bash -n`, `py_compile`, shellcheck adds none): OK.
- `mutants.sh`: patched ALL PASS (5 checks); without 0016 it fails (5 checks FAIL); 10 of 10 mutants killed; `exit 0`
  (log `~/QwenPRs/logs/mutants-qwen-pr-mtp-graphs.log`). Passes with all seven patches applied. The first run had one
  survivor (`image-graphed`): the check built a new stream object on every call, so the key never repeated; it now
  reuses the streams and kills it.
- Leak scan of `pr.diff`: nothing.
- Evidence: `scripts/qwen38/mtp-graph/README.md` (status, the in-process table, the GPU test) and
  `scripts/qwen38/ple-defer/README.md` (the HTTP A/B table, "production" and "mtp-graph" columns).

Awaiting the owner's approval of the text below.

---

**Title:** Patch 0016: the MTP head's chain steps as CUDA graphs: decode +0.3 to +1.4% through the server, the eager step's bits

**Body:**

## Description and Motivation

A decode round drafts with up to `MTP_DRAFTS` MTP head steps. After the round's absorb, each step runs one row per chaining stream. Eager, such a step is ~40 kernel launches from Python, and its host time is close to its GPU time, so the GPU waits between steps. A trace of one stream's steady decode showed the GPU idle for 12% of it, with 15-60 us gaps before each kernel of an MTP step.

This patch adds `ChainGraphs`. A step of one row per stream, with no image stream, replays a CUDA graph of `mtp_compute`:

- Key: the stepping streams' MTP cache addresses (the attention's captured pointer table) and a power-of-two context bucket, at least 8,192. The attention's sparse select is launched for every stream with the bucket as its context. Its kernels read each row's real end from the device, so the extra blocks and chunks score, select and merge nothing, and a replay computes the eager step's bits.
- Per replay, only the rows' positions are uploaded, one pinned copy of the step's integer table.
- A key is captured the second time it is seen, so a one-off set of chaining streams stays eager. At most 32 graphs; the least recently replayed goes first.
- Steps of more than one row a stream, and image streams, stay eager.

`TENSORFOLD_MTP_GRAPHS=0` runs every step eagerly.

The bits do not change: a replay computes what the eager step computes, so the drafts, and therefore which drafts land, are the same. The replies are verified as before.

Measured on one DGX Spark through the server, interleaved A/B, 3 passes an arm, arm order rotated, cold n-gram table every boot, median tok/s a request:

| Workload | 1 stream | 2 streams | 4 streams |
| --- | ---: | ---: | ---: |
| code, before | 90.2 | 63.6 | 42.1 |
| code, after | 91.3 (+1.2%) | 64.5 (+1.4%) | 42.4 (+0.7%) |
| prose, before | 74.6 | 51.0 | 32.2 |
| prose, after | 75.2 (+0.8%) | 51.5 (+1.0%) | 32.3 (+0.3%) |

Prefill was unchanged (8k / 32k / 128k: 2,548 / 2,634 / 2,301 -> 2,543 / 2,638 / 2,298 tok/s). In-process (steady-state decode, 3 passes x 2 windows an arm, interleaved), against the same code without it, it measured +0.7 to +2.1% a stream in all six cells. With #<0017 PR> on top, the gain through the server is +1.8 to +2.2% in every cell.

`0016` is a placeholder. This PR depends on #<0015 PR> (its `multi.py` hunks sit on 0015's), and so on #<0012 PR> and #<0013 PR>. #<0017 PR> builds on it.

## Related Issues

None.

---

## Testing

Setup: one DGX Spark (GB10), DGX OS, the recipe at `4c0dea8` with its defaults (`PARALLEL=5`, `CONTEXT=262144`, int8 KV, `PLE_ON_SSD=1`, vision on) except, in both arms, my patches 0012-0015 and 0018 with `TENSORFOLD_CHAIN_RULE=cum` and `MTP_DRAFTS=8`. Patches applied with the recipe's own `patch -p0 --forward` onto the recipe image's site-packages (after 0002); I did not run `scripts/prepare.sh` itself.

- `git apply` at `4c0dea8`, `bash -n start.sh stop.sh scripts/*.sh`, `python3 -m py_compile tools/*.py`: pass. shellcheck finds nothing new.
- Bits: six fixed prompts at temperature 0 (seed 1234, thinking off) give byte-identical full replies to the recipe's own image (6/6), and the A/B's 8 decode replies (greedy) were identical across arms and passes.
- On the GPU, with the server stopped, my test ran production's eager step before every replay on the same state, kept its draft logits and the state it writes, poisoned that state and the step's attention scratch, and required the replay's logits and written state to be bit-identical: 3,205 replays, at 1, 2 and 4 streams, including one key whose contexts crossed the 512-token chunks and the 256-token score grid, and a decode crossing the 8,192 bucket. The same requests with the graphs off and on gave identical replies, rounds and accepted drafts. That test needs the GPU and the model, so it is not in this PR.
- `tools/mtp_graph_check.py` (CPU, no model) runs `ChainGraphs` with CUDA graphs stood in for by recorders whose replay re-runs the captured kernel on the tensors it captured, and `mtp_compute` by a kernel that writes each row's result from the device-side position tables. It checks: eager at a key's first sight, then replays whose results match the streams' current positions over 2,900 steps of one key; a new bucket past 8,192; multi-row and image steps stay eager; the 32-graph limit drops the least recently replayed; `MultiDecoder._mtp` uses it and clears the step's tables; `TENSORFOLD_MTP_GRAPHS=0` turns it off. It passes with the patch (5 checks), fails without it (5 of 5), and fails on each of 10 mutants: stale positions, stale block starts, the table not uploaded, an exact instead of bucketed context, the bucket missing from the key, multi-row steps graphed, image steps graphed, capture at first sight, the LRU order not refreshed, the graphs off by default.
- In production on our Spark since 2026-10-08.
- Memory: the startup estimate is unchanged (our server's log: 85.81 GiB within 115.52 GiB, with this patch). The graphs share one memory pool and replay into the MTP head's existing buffers; I did not measure the pool's size separately. Our server has run with it at 5 x 262,144 since 2026-10-08.

---

## Checklist:

- [x] I have read the README and `scripts/config.sh` and kept my changes consistent with them.
- [x] My pull request has a sound title and description (not something vague like `Update README.md`).
- [x] My change is reproducible and verified (script syntax check, `scripts/prepare.sh`, a launch, or a re-measurement).
- [x] A patch change keeps every reply byte-identical, and I said how I checked it.
- [x] I updated the README and/or `scripts/config.sh` if a setting, default, or measured number changed.
- [x] If my change affects memory, I checked the startup estimate still fits the budget at the default 5 x 262,144 int8 setting.
- [x] Defaults in `scripts/config.sh` still work out of the box; a new setting has a sane fallback like the existing ones.
