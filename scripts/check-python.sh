#!/bin/sh
# Static check for the python in this repo (the Qwen serve's layers, harnesses and tests, and the upstream tools). Motivated by a real defect in
# ots-infra: an edit removed a helper (mkdirs) from og-dispatch.py while a call site remained -
# python only fails at runtime, so a broken tool shipped and was caught only when it ran.
#
# Gating classes (a defect here means broken code): syntax errors, undefined names.
# Non-gating (reported as INFO): unused imports/variables, f-string nits - the repo carries
# dozens of those in one-off research probes and gating on them would bury the real signal.
#
# Exit: 0 clean/info-only, 1 severe finding, 0 SKIPPED if pyflakes is unavailable.
set -u

# Use the first interpreter that can actually import pyflakes. `python3` first-only picked the
# Microsoft Store stub (WindowsApps\python3.exe) on Windows and SKIPPED although the real python had
# pyflakes installed - a gate that skips when its tool is present checks nothing.
PY=""
for c in python3 python; do
  p=$(command -v "$c") || continue
  "$p" -c "import pyflakes" 2>/dev/null && { PY=$p; break; }
done
if [ -z "$PY" ]; then
  echo "check-python: SKIPPED - pyflakes not installed (pip install pyflakes)"
  exit 0
fi

FILES=$(find scripts upstream -name '*.py' 2>/dev/null | sort)
[ -n "$FILES" ] || { echo "check-python: no python files in scope"; exit 0; }
N=$(echo "$FILES" | wc -l | tr -d ' ')

# shellcheck disable=SC2086
OUT=$($PY -m pyflakes $FILES 2>&1 || true)

SEVERE=$(echo "$OUT" | grep -E "undefined name|was never closed|invalid syntax|unexpected (EOF|indent)|syntax error" || true)
INFO=$(echo "$OUT" | grep -vE "undefined name|was never closed|invalid syntax|unexpected (EOF|indent)|syntax error" | grep -E ":[0-9]+:[0-9]+:" || true)

info_n=$(echo "$INFO" | grep -c . 2>/dev/null || echo 0)
if [ "$info_n" -gt 0 ]; then
  echo "== info (non-gating style findings, $info_n) =="
  echo "$INFO" | sed 's/^/  /'
fi

if [ -n "$(echo "$SEVERE" | grep .)" ]; then
  echo "== SEVERE (gating) =="
  echo "$SEVERE" | sed 's/^/  /'
  echo "check-python: FINDINGS across $N file(s)"
  exit 1
fi
echo "  ok: no severe findings across $N file(s)"
echo "check-python: CLEAN"
