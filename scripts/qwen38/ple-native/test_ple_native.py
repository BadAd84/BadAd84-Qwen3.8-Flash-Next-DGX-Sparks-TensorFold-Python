#!/usr/bin/env python3
"""Test of ple-native (ssd_table.py's native reader) on the checkpoint's real n-gram table, CPU only: every gather
through the native reader against the Python reads of the same module (TensorFold v0.6.1 + ple-cache: the oracle),
bit for bit (words, scales, biases: dtype, shape and bytes), over id sets shaped like the engine's: a decode window's
(tens to thousands of ids, duplicates), a prompt chunk's (16,384 ids), long runs of adjacent rows (one read covers
many rows, split at MAX_READ), every shard boundary, unsorted with repeats; with the table's pages dropped first
(cold, the disk) and cached (warm). Also: the native reader is built and engaged; then each path's time.

Prints PASS/FAIL lines and ``exit N``. Usage: python3 test_ple_native.py
"""
from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

FAILS: list[str] = []


def fail(msg: str) -> None:
    FAILS.append(msg)
    print(f"FAIL {msg}", flush=True)


def table():
    from tensorfold.families.qwen4_exp import host_table

    model_dir = Path(glob.glob("/root/.cache/huggingface/hub/models--Vontra--Qwen3.8-Flash-Next-MLX-4bit-MTP/"
                               "snapshots/*/")[0])
    keys = set()
    for path in sorted(model_dir.glob("model*.safetensors")):
        keys |= set(host_table.read_header(path))
    shards = sorted({k.rsplit(".shard_", 1)[0] for k in keys if ".ngram_embedding.shard_" in k})
    if len(shards) != 1:
        raise SystemExit(f"expected one n-gram table, found {shards}")
    name = shards[0]
    count = len({k.rsplit(".", 1)[0] for k in keys if k.startswith(name + ".shard_") and k.endswith(".weight")})
    return host_table.from_checkpoint(model_dir, name, count, ssd=True)


def evict(t) -> None:
    for fd in t._fds:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def main() -> None:
    from tensorfold.families.qwen4_exp import ssd_table

    t = table()
    if not isinstance(t, ssd_table.SSDTable):
        fail(f"the table is a {type(t).__name__}, not an SSDTable")
    if ssd_table._native() is None or not ssd_table.NATIVE:
        fail("the native reader is not built or not engaged (NATIVE / _native())")
    print(f"table: {t.rows} rows in {len(t.starts) - 1} shards over {len(t._fds)} files, row {t.wrow} + 2 x {t.grow} B",
          flush=True)
    rng = np.random.default_rng(1234)
    bounds = np.concatenate([t.starts[1:-1] - 1, t.starts[1:-1], [0, t.rows - 1]])
    cases = [("decode-1", rng.integers(0, t.rows, 8)),
             ("decode-1c", rng.integers(0, t.rows, 112)),
             ("decode-4c", np.concatenate([rng.integers(0, t.rows, 400), rng.integers(0, t.rows, 30).repeat(3)])),
             ("chunk", rng.integers(0, t.rows, 16384)),
             ("runs", np.concatenate([np.arange(s, s + n) for s, n in zip(rng.integers(0, t.rows - 20000, 6),
                                                                           (1, 2, 50, 700, 9000, 20000))])),
             ("bounds", np.concatenate([bounds, bounds[::-1], bounds + 1, np.maximum(bounds - 1, 0)])),
             ("unsorted-repeats", rng.permutation(np.repeat(rng.integers(0, t.rows, 3000), 4)))]
    for cold in (True, False):
        for name, ids in cases:
            ids = np.clip(ids, 0, t.rows - 1).reshape(-1, 8) if ids.size % 8 == 0 else np.clip(ids, 0, t.rows - 1)
            if cold:
                evict(t)
            ssd_table.NATIVE = True
            t0 = time.perf_counter()
            got = t.gather(ids)
            t_native = time.perf_counter() - t0
            if cold:
                evict(t)
            ssd_table.NATIVE = False
            t0 = time.perf_counter()
            ref = t.gather(ids)
            t_python = time.perf_counter() - t0
            ssd_table.NATIVE = True
            ok = all(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b) for a, b in zip(got, ref))
            line = (f"{'cold' if cold else 'warm'} {name}: {ids.size} ids, native {t_native * 1e3:.2f} ms, "
                    f"python {t_python * 1e3:.2f} ms, outputs {'identical' if ok else 'DIFFERENT'}")
            if ok:
                print(f"PASS {line}", flush=True)
            else:
                fail(line)
    t.close()
    code = 1 if FAILS else 0
    print(f"{'FAIL' if FAILS else 'PASS'}: {len(FAILS)} failures", flush=True)
    print(f"exit {code}", flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
