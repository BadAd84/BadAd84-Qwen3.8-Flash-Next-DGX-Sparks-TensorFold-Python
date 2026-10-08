# qwen3.8-flash-next-dev

Development of OTS's Qwen3.8 Flash Next serve on spark4 (`:8888`, one Spark): the MiaAI-Lab recipe, our layers on top
of it, their tests and benchmarks, and the upstream PRs. Split out of ots-infra on 2026-10-08 (the history up to the
split stays there: ots-infra `git log -- scripts/qwen38 upstream/qwen-pr-*`).

The hardware - the Sparks, provisioning, what runs where - is ots-infra's (`docs/INFRA-LAYOUT.md`,
`docs/NEW-SPARK-SETUP.md`). The GLM-5.3-Flash serve on spark1-3 is glm5.3-flash-dev.

## Now (2026-10-08)
- **Production (spark4):** the recipe @ `4c0dea8` (TensorFold's Python engine v0.6.1, the MLX 4-bit checkpoint
  `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`) + seven OTS layers, image `tensorfold-qwen38:v061-ots`, started by
  `qwen-start.sh`.
- **Upstream moved:** the recipe's release 1.0.0 (main @ `364f21a`) is on TensorFold's Zig engine with the
  INT4-AutoRound checkpoint and FP8 KV. It runs on spark6 for a two-box comparison against production
  (`scripts/qwen38/zig2_ab.sh`). Our seven upstream packages target the Python recipe and are PARKED.

## docs/
- **`SPEED-CAMPAIGN-2026-10.md`** - where the serve spends its time (profile, attribution), the layers kept and the
  ideas dropped, each with its HTTP A/B (golden 6/6 byte-identical).
- **`UPSTREAM-WATCHLIST.md`** - the recipe and engine upstream against what spark4 runs, the Zig move, and which
  checkpoint formats each engine reads.

## scripts/qwen38/
The serve: `qwen-start.sh`, `build_qwen_ots.sh`, the seven layers (`pr319/`, `ple-cache/`, `ple-native/`,
`chain-stop/`, `mtp-graph/`, `ple-defer/`, `tokenize-nogil/`, each with its test and mutant runner), the A/B runners
(`chain_http_ab.sh`, `stock_ab.sh`, `zig_ab.sh`, `zig2_ab.sh`) and their summaries. `scripts/qwen38/README.md` holds
every layer's evidence.

## upstream/
The seven recipe PR packages (`qwen-pr-*`: `pr.diff`, `PR.md`, `mutants.sh`), parked; `BATCH-2026-10.md` and
`compile_batch.py` (writes `BATCH-2026-10-TEXTS.md`).
