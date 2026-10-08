#!/usr/bin/env python3
"""GPU test of hc-split (qmm.hc_upmix launching ``_qmm_upmix4``, the four streams in one dot):

1. synthetic: production-shaped random 4-bit up weights (N 10240, K 320), rows 1-16, inputs at several scales plus
   exact zeros and negative zeros (a sum that starts from v0 instead of zero shows only there): ``hc_upmix`` with
   ``HC_WIDE`` on must equal v0.6.1's ``_qmm_upmix`` (``HC_WIDE`` off) bit for bit, outputs pre-filled with NaN.
2. the engine built exactly as the serve builds it, test_mtp_graph.py's requests: every eager ``hc_upmix`` call of the
   serve (real weights, real activations; calls inside a CUDA graph capture are not compared) is run both ways on
   NaN-filled copies of its outputs and must be bit-identical (the first difference ends the process).
3. the same requests with the new kernel (and the MTP graphs as built) and with v0.6.1's kernel (graphs off, so no
   graph captured with the new kernel replays): every reply's tokens, rounds and accepted drafts identical.

Prints PASS/FAIL lines and ``exit N``; exit 1 on any FAIL. Usage: python3 test_hc_split.py [count]
(test_mtp_graph.py importable beside it).
"""
from __future__ import annotations

import sys
import traceback
from collections import Counter

import torch

import test_mtp_graph as T

FAILS: list[str] = []
D, S, LOW = 2560, 4, 320
WIDE = D * S


def fail(msg: str) -> None:
    FAILS.append(msg)
    print(f"FAIL {msg}", flush=True)


def both(qmm, act, xs_act, q, normed, streams, upmix=None):
    """(new mixed, new sums), (old mixed, old sums): ``hc_upmix`` (``upmix``: the unpatched one) with HC_WIDE on and
    off, outputs NaN-filled."""
    upmix = upmix or qmm.hc_upmix
    m, d = act.shape[0], q.n // streams
    outs = []
    for wide in (True, False):
        mixed = torch.full((m, d), float("nan"), device=act.device, dtype=torch.bfloat16)
        sums = torch.full((m, d // 32), float("nan"), device=act.device, dtype=torch.float32)
        qmm.HC_WIDE = wide
        try:
            upmix(act, xs_act, q, normed, mixed, sums, streams)
        finally:
            qmm.HC_WIDE = True
        outs.append((mixed, sums))
    return outs


def same(a, b) -> bool:
    return torch.equal(a[0].view(torch.int16), b[0].view(torch.int16)) and torch.equal(a[1].view(torch.int32),
                                                                                         b[1].view(torch.int32))


def synthetic(qmm) -> None:
    def q4(seed: int):
        g = torch.Generator(device="cuda").manual_seed(seed)
        w = torch.randint(-2**31, 2**31 - 1, (WIDE, LOW // 8), dtype=torch.int32, device="cuda", generator=g)
        s = ((torch.rand((WIDE, LOW // 32), device="cuda", generator=g) - 0.5) * 0.02).to(torch.bfloat16)
        b = ((torch.rand((WIDE, LOW // 32), device="cuda", generator=g) - 0.5) * 0.02).to(torch.bfloat16)
        return qmm.make_q4(w, s, b, "tiled")

    ups = [q4(300 + i) for i in range(3)]
    checked = bad = 0
    for m in range(1, 17):
        g = torch.Generator(device="cuda").manual_seed(m)
        for trial in range(6):
            act = (torch.randn((m, LOW), device="cuda", generator=g) * (1 + 3 * trial)).to(torch.bfloat16)
            normed = torch.randn((m, WIDE), device="cuda", generator=g).to(torch.bfloat16)
            if trial == 4:
                act[:, ::3] = 0
                normed[:, ::5] = 0
            if trial == 5:                                   # every stream's product -0: the mix's sign of zero
                normed[:, ::2] = -0.0
                normed[m // 2:, :] = -0.0
            new, old = both(qmm, act, qmm.group_sums(act), ups[trial % 3], normed, S)
            torch.cuda.synchronize()
            checked += 1
            if not same(new, old):
                bad += 1
                print(f"FAIL synthetic R={m} trial {trial}: the four-stream kernel's bits differ", flush=True)
    line = f"synthetic: {checked} launches (rows 1-16, 6 input sets) checked against v0.6.1's kernel, {bad} different"
    if bad:                                              # no engine needed to know: end now
        T.die(line)
    print(f"PASS {line}", flush=True)


def main() -> None:
    from tensorfold import hub
    from tensorfold.cuda.server import App
    from tensorfold.families.qwen4_exp import cuda_engine
    from tensorfold.families.qwen4_exp.cuda import qmm

    if not qmm.HC_WIDE:
        T.die("the four-stream kernel is off in the module as imported (TENSORFOLD_HC_WIDE)")
    synthetic(qmm)
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    model_dir = hub.resolve("Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP", download=False)
    engine = cuda_engine(model_dir, drafter="", tp=1, rank=0, master="", master_port=29551, no_drafts=False,
                         mtp_drafts=8, mtp_confidence=0.60, context=262144, context_explicit=True,
                         ple_on_ssd=True, kv_dtype="int8", parallel=5, vision=True, vision_urls=False)
    app = App(engine, model_dir, "Qwen3.8-Flash-Next", default_thinking=True,
              sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 20}, max_tokens=32768,
              context_window=262144, vision_max_images=50)
    md = engine.multi
    graphs = getattr(md, "chain_graphs", None)
    orig = qmm.hc_upmix
    rows = Counter()

    def checked(act, xs_act, q, normed, mixed, xs_mixed, streams):
        try:
            if qmm.HC_WIDE and not torch.cuda.is_current_stream_capturing():
                new, old = both(qmm, act, xs_act, q, normed, streams, orig)
                if not same(new, old):
                    T.die(f"an up mix of {act.shape[0]} rows: the four-stream kernel's bits differ from v0.6.1's")
                rows[act.shape[0]] += 1
            return orig(act, xs_act, q, normed, mixed, xs_mixed, streams)
        except BaseException:
            traceback.print_exc()
            T.die("the up-mix check raised")

    for name, clients, prompts in T.TESTS:
        prompts = T.prompts_of(app, name, prompts)
        n = T.count_of(name, count)
        eos = name not in T.LONG
        msgs = [f"{p}\n\n(test {name}-{i})" for i, p in enumerate(prompts[:clients])]
        rows.clear()
        qmm.hc_upmix = checked
        while md._evict_kept(None):
            pass
        on = T.run(engine, app, msgs, n, eos)
        qmm.hc_upmix = orig
        line = f"{name}: {sum(rows.values())} serve up mixes checked, all identical; rows {dict(sorted(rows.items()))}"
        fail(line) if not rows else print(f"PASS {line}", flush=True)
        qmm.HC_WIDE = False                                  # v0.6.1's kernel, eagerly
        if graphs is not None:
            md.chain_graphs = None
        while md._evict_kept(None):
            pass
        try:
            off = T.run(engine, app, msgs, n, eos)
        finally:
            qmm.HC_WIDE = True
            if graphs is not None:
                md.chain_graphs = graphs
        for i, (a, b) in enumerate(zip(on, off)):
            line = (f"{name} stream {i}: four-stream {a['sha']} {a['n']} tok {a['rounds']} rounds {a['accepted']} "
                    f"accepted, v0.6.1 {b['sha']} {b['n']} tok {b['rounds']} rounds {b['accepted']} accepted")
            fail(line) if a != b else print(f"PASS {line}", flush=True)
    engine.close()
    code = 1 if FAILS else 0
    print(f"{'FAIL' if FAILS else 'PASS'}: {len(FAILS)} failures", flush=True)
    print(f"exit {code}", flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
