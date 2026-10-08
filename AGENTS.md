# qwen3.8-flash-next-dev - repo rules, conventions, and gotchas

The Qwen3.8 Flash Next serve on spark4 as a development project: our layers on the MiaAI-Lab recipe, their tests and
benchmarks, and the upstream PRs. The hardware is ots-infra's; the GLM serve is glm5.3-flash-dev.

## Layout
- `scripts/qwen38/` - the serve's files, AUTHORED here and COPIED to spark4 (`~/qwen-start.sh`, `~/qwen-ots/` for the
  image build, `~/work/` for the harnesses); the repo copy is the source. `scripts/qwen38/README.md` is the SSOT for
  the serve and each layer's evidence.
- `docs/` - the speed campaign and the upstream watchlist.
- `upstream/` - recipe PR packages; skill `submit-ots-patch-as-miaai-recipe-pr`.

## Conventions
- **GPG-signed commits. Never `--no-gpg-sign`** (pinentry = YubiKey touch; if it fails, retry).
- **Current-state docs**; retired things marked RETIRED in one line; "current" claims carry a date.
- **No em-dashes** - plain hyphen. Never `git add -A`; stage explicit paths. New `.sh` files need
  `git update-index --chmod=+x`.
- Secrets never enter this repo (or a prompt). Fleet auth is YubiKey/agent-based.
- **A change ships only with replies byte-identical to the image it replaces** (golden 6/6:
  `~/work/qf_harness.py golden`, then `diff`), and its speed from an interleaved HTTP A/B on fixed prompts (3 passes
  an arm, the order alternating), never a single run: decode varies up to ~25% run to run on this box.

## Gate
- `scripts/check-python.sh` - pyflakes over `scripts/` and `upstream/`. GATES on syntax errors and undefined names
  only; style findings are INFO.

## Gotchas (each cost real time)
- `build_qwen_ots.sh` must build from LF files: five layer files once went into the image with CRLF (it strips CRs
  now).
- The n-gram tables page from disk on the Python engine: a cold boot measures slower than a warm one. A/Bs boot each
  arm from a cold table (`ple_evict.py`).
- A server that keeps prompts answers a repeated prompt from its cache: qf_run.py's prefill prompts repeat every pass,
  so cold prefill needs prompts new to the server (`prefill_fresh.py`; the Zig release answered a repeated 128k
  prompt in 0.31 s).
- spark4 is stock DGX OS on a 1 Gb/s LAN link; spark6 (the comparison box) is the same. Production restarts take
  ~2 min; the container restarts by itself after a reboot (`unless-stopped`).
- Nesting heredocs inside `ssh '...'` breaks quoting: write a script, `scp` it, run it (or base64 it).
