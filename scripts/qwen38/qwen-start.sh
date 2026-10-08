#!/usr/bin/env bash
# qwen-start.sh (spark4, copied to ~): start or restart the Qwen3.8 Flash Next serve on :8888 with the OTS image
# (tensorfold-qwen38:v061-ots = the recipe's image + TensorFold PR #319 + ple-cache + chain-stop, build_qwen_ots.sh).
# A plain `./start.sh` in ~/qwen38-flashnext would start the recipe's own image and drop those. Arguments go to the
# recipe's start.sh (e.g. `qwen-start.sh restart`). The container restarts by itself after a reboot (restart policy
# unless-stopped).
set -euo pipefail
export IMAGE=tensorfold-qwen38:v061-ots
# MTP chains: up to 8 drafts a round, a chain stopping once the calibrated chance that all its drafts land falls under
# 0.30 (README.md, chain-stop). Today's recipe rule instead: MTP_DRAFTS=6 and TENSORFOLD_CHAIN_RULE=conf.
export MTP_DRAFTS="${MTP_DRAFTS:-8}"
export TENSORFOLD_CHAIN_RULE="${TENSORFOLD_CHAIN_RULE:-cum}"
export TENSORFOLD_CHAIN_T="${TENSORFOLD_CHAIN_T:-0.30}"
cd "$HOME/qwen38-flashnext"
./start.sh "$@" </dev/null
docker update --restart unless-stopped qwen38-flash-next-tf >/dev/null
