# Qwen3.8 Flash Next on one DGX Spark - upstream watchlist

Current state of the upstream recipe and engine against what spark4 runs. Live config: `scripts/qwen38/README.md`.

## 2026-10-08

- Live (spark4): the MiaAI-Lab recipe `Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold` @ `4c0dea8` (TensorFold's Python
  engine v0.6.1, the MLX 4-bit checkpoint `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`) + seven OTS layers, image
  `tensorfold-qwen38:v061-ots` (`scripts/qwen38/`).
- The recipe moved to TensorFold's Zig engine on 2026-10-08 (release 1.0.0, main @ `364f21a`: TensorFold `db281878`
  + 9 patches, image `zig-db28187-4144696f21e3`; the INT4-AutoRound checkpoint
  `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`, FP8 KV, the n-gram table in memory). Our seven packages
  (`upstream/qwen-pr-*/`, `upstream/BATCH-2026-10.md`) patch the Python recipe it replaced and no longer apply to
  main; they are PARKED (owner 2026-10-08).
- Checkpoint formats, from the source (2026-10-08): the Python engine (v0.6.1 to the last Python release v0.6.6) reads
  this model as MLX 4-bit, EXL3 or NVIDIA ModelOpt NVFP4, and has no GPTQ / AutoRound code at all; the Zig engine on
  CUDA reads GPTQ 4-bit (symmetric, groups of 128, no act order) or the ModelOpt NVFP4 "MIXED_PRECISION" export, and
  refuses MLX (recipe `patches/0003-config-weights.patch`, `checkGptq` and the `.modelopt` check). The one checkpoint
  both read is `nvidia/Qwen3.8-Flash-Next-NVFP4` (123.6 GiB); MiaAI-Lab's `nvfp4-v066` branch measured it on the
  Python engine v0.6.6: prose / code 35.2 / 52.1 tok/s at one request, prefill 1,479-1,616 tok/s, against 2,180-2,468
  for the MLX checkpoint (its 47.7 GiB FP8 n-gram table pages from disk).
- The Zig release serves on spark6 for the two-box comparison against spark4 (`scripts/qwen38/zig2_ab.sh`; it is
  also staged, not serving, in spark4's `~/qwen-zig`). Measured 2026-10-08 (`scripts/qwen38/README.md`): code decode
  1.14-1.30x production at 1-4 clients, prose 0.82x at one client and 1.13x at four; fresh prefill 1.07-1.08x, which
  spark6's ~8% compute lead accounts for.
- TensorFold's Python engine is frozen at v0.6.6 (`python-0.6` branch); TensorFold's own 1.0.x release builds serve
  only Nemotron 3.5 on a GB10 (`capabilities --json`), so this recipe's Zig build is TensorFold's `zig-flashnext` line
  plus the recipe's patches.
- Automated: Hermes cron `Upstream watch` (every 6 h, ots-infra `scripts/ct106/upstream_watch.py`) messages the
  Telegram status topic when this recipe's main moves or a TensorFold release appears.
