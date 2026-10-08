#!/usr/bin/env python3
"""GPU test of ple-defer (forward.stage(defer_ple=True), run from forward.ple_block; MultiDecoder.round uses it for a
decode round alone): the engine built exactly as the serve builds it, test_mtp_graph.py's requests (1, 2 and 4
concurrent streams, a ~1.4k-token and a ~7.9k-token prompt), then:

1. every decode round's n-gram rows: after the round's forward, the rows its staging copied to the GPU
   (``ple_w`` / ``ple_s`` / ``ple_b``) must equal, bit for bit, the table's rows of the ids ``stage`` looks up (read
   again on the host); a round that deferred its staging must have had it run by the n-gram layer (never lost), and
   the deferral must have engaged in the test. A test with no deferred round FAILs.
2. the same requests with the deferral off (``multi.PLE_DEFER = False``, v0.6.1's staging) and on: every reply's
   tokens identical, every stream's rounds and accepted drafts identical.

A failed row check, or any exception in the round hook, prints its FAIL line and ends the process at once (exit 1).
Prints PASS/FAIL lines and ``exit N``; exit 1 on any FAIL. Usage: python3 test_ple_defer.py [count]
(test_mtp_graph.py importable beside it).
"""
from __future__ import annotations

import functools
import sys
import traceback

import numpy as np
import torch

import test_mtp_graph as T

FAILS: list[str] = []


def fail(msg: str) -> None:
    FAILS.append(msg)
    print(f"FAIL {msg}", flush=True)


def main() -> None:
    from tensorfold import hub
    from tensorfold.cuda.server import App
    from tensorfold.families.qwen4_exp import cuda_engine
    from tensorfold.families.qwen4_exp.cuda import forward, multi

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    model_dir = hub.resolve("Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP", download=False)
    engine = cuda_engine(model_dir, drafter="", tp=1, rank=0, master="", master_port=29551, no_drafts=False,
                         mtp_drafts=8, mtp_confidence=0.60, context=262144, context_explicit=True,
                         ple_on_ssd=True, kv_dtype="int8", parallel=5, vision=True, vision_urls=False)
    app = App(engine, model_dir, "Qwen3.8-Flash-Next", default_thinking=True,
              sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 20}, max_tokens=32768,
              context_window=262144, vision_max_images=50)
    md = engine.multi
    buf = md.buf
    stats = {"rounds": 0, "deferred": 0, "lost": 0, "checked": 0, "same": 0}
    pending = {"set": False}
    staged: list = []                                    # (layer, ids, at) of the current round's staging
    orig_rows = forward.stage_ple_rows
    orig_stage = multi.stage
    orig_round = multi.MultiDecoder.round

    def rows_(p, b, ids, at=0):
        if b is buf:
            staged.append((p, np.array(ids, copy=True), at))
        return orig_rows(p, b, ids, at)

    def stage_(w, b, windows, **kw):
        segs = orig_stage(w, b, windows, **kw)
        if b is buf and kw.get("defer_ple") and getattr(b, "ple_staging", None) is not None:
            stats["deferred"] += 1
            pending["set"] = True
        return segs

    @functools.wraps(orig_round)
    def round_(self):
        staged.clear()
        pending["set"] = False
        out = orig_round(self)
        try:
            if pending["set"] and not staged:            # deferred, then never staged by the n-gram layer
                stats["lost"] += 1
            if staged:
                stats["rounds"] += 1
                torch.cuda.synchronize()
                for p, ids, at in staged:
                    words, scales, biases = p.table.gather(ids)
                    n = words.shape[0]
                    got = (buf.ple_w[at:at + n].cpu().numpy(),
                           buf.ple_s[at:at + n].view(torch.int16).cpu().numpy(),
                           buf.ple_b[at:at + n].view(torch.int16).cpu().numpy())
                    stats["checked"] += 1
                    if not (np.array_equal(got[0].view(np.uint32), words.reshape(n, -1).view(np.uint32))
                            and np.array_equal(got[1].view(np.uint16), scales.reshape(n, -1))
                            and np.array_equal(got[2].view(np.uint16), biases.reshape(n, -1))):
                        T.die(f"the n-gram rows at {at} ({n} rows) differ from the table's rows of their ids")
                    stats["same"] += 1
        except BaseException:
            traceback.print_exc()
            T.die("the round check raised")
        return out

    forward.stage_ple_rows = rows_
    multi.stage = stage_
    multi.MultiDecoder.round = round_
    for name, clients, prompts in T.TESTS:
        prompts = T.prompts_of(app, name, prompts)
        n = T.count_of(name, count)
        eos = name not in T.LONG
        msgs = [f"{p}\n\n(test {name}-{i})" for i, p in enumerate(prompts[:clients])]
        for k in stats:
            stats[k] = 0
        multi.PLE_DEFER = True
        while md._evict_kept(None):
            pass
        on = T.run(engine, app, msgs, n, eos)
        line = (f"{name}: {stats['rounds']} decode rounds staged, {stats['deferred']} deferred, {stats['lost']} "
                f"lost; {stats['checked']} staged row sets checked against the table, {stats['same']} identical")
        if stats["deferred"] == 0 or stats["lost"] or stats["checked"] == 0 or stats["same"] != stats["checked"]:
            fail(line)
        else:
            print(f"PASS {line}", flush=True)
        multi.PLE_DEFER = False
        while md._evict_kept(None):
            pass
        off = T.run(engine, app, msgs, n, eos)
        for i, (a, b) in enumerate(zip(on, off)):
            line = (f"{name} stream {i}: deferred {a['sha']} {a['n']} tok {a['rounds']} rounds {a['accepted']} "
                    f"accepted, staged first {b['sha']} {b['n']} tok {b['rounds']} rounds {b['accepted']} accepted")
            if a != b:
                fail(line)
            else:
                print(f"PASS {line}", flush=True)
    engine.close()
    code = 1 if FAILS else 0
    print(f"{'FAIL' if FAILS else 'PASS'}: {len(FAILS)} failures", flush=True)
    print(f"exit {code}", flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
