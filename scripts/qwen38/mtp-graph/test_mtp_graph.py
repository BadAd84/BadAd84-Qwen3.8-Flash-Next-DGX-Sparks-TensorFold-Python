#!/usr/bin/env python3
"""GPU test of mtp-graph (ChainGraphs in multi.py): the engine built exactly as the serve builds it, greedy requests at
1, 2 and 4 concurrent streams on prose and code prompts, one ~1.4k-token prompt decoded 1,100 tokens (``1M``: inside
one graph key its context crosses the 512-token attention chunks at 1536 and 2048, the sparse onset at 2052 and the
256-token score grid at 2304) and one ~7.9k-token prompt whose decode crosses the 8192 bucket (``1L``), then:

1. every graph replay against production's eager MTP step on the same state: before each replay the eager step (a
   fresh ``attn_multi.Step``, unbucketed: v0.6.1's path) runs and its draft-head logits and the state it writes are
   kept (each stream's MTP cache row at its MTP length: k, v and their scales, the indexer key, and the pooled block
   that row completes). Then that state and the attention scratch the step fills (chunk partials, key lists, key
   counts, block scores) are POISONED, so the replay must compute every byte itself and nothing the eager step left
   behind can mask a missing launch or a stale table; the replay's logits and the state it writes must be
   bit-identical (``torch.equal`` on integer views). A test with no replay FAILs, and the ``1M`` test FAILs unless a
   replay ran past its graph's capture context across a chunk step and a score-grid step.
2. no step of more than one row a stream may run as a graph (``ChainGraphs.run`` must return None for it).
3. the same requests with the graphs off (``chain_graphs = None``), on with the checks, and on WITHOUT the checks
   (production's path, nothing eager in between): every reply's tokens identical, and every stream's rounds and
   accepted drafts identical (identical drafts).

A check that fails, or any exception in a graph step, prints its FAIL line and ends the process at once (exit 1): an
engine thread that died would otherwise leave the requests waiting forever.
Prints PASS/FAIL lines and ``exit N``; exit 1 on any FAIL.
Usage (in the image, the patched multi.py mounted): python3 test_mtp_graph.py [count]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import traceback

import torch

PROSE = ["Explain how a lock-free ring buffer passes messages between one writer and one reader without a mutex.",
         "Write a few paragraphs on why weather forecasts get worse further into the future, using sensitive "
         "dependence rather than a list of model names.",
         "Explain the difference between TCP and UDP congestion control to a new graduate engineer.",
         "Describe how a librarian merges a cart of call-number-sorted books into the stacks."]
CODE = ["Write a Python quicksort implementation with a docstring and unit tests.",
        "Write a Python function that merges two sorted linked lists, with type hints and tests.",
        "Write a Rust struct for a bounded ring buffer with push and pop methods and tests.",
        "Write a Python context manager that retries a block on transient IO errors, with tests."]
TESTS = (("1c", 1, CODE), ("1p", 1, PROSE), ("2c", 2, CODE), ("4p", 4, PROSE),
         ("1M", 1, None), ("1L", 1, None))
# MTPG_TESTS=1M,1L runs only those tests (diagnosis; the mutant runners run them all)
ONLY = [t for t in os.environ.get("MTPG_TESTS", "").split(",") if t]
# the random-word tests: prompt tokens, tokens decoded (None: the command line's count; decoded to the count), and
# the request after the words (1M asks for new prose: a summary of random words soon repeats and copy drafts take
# over its rounds, leaving no chain steps past ~1.9k)
LONG = {"1M": (1400, 1100, "Ignore those notes. Write a long, detailed essay of at least 1,500 words on the history "
                           "of bridge engineering, from timber beams to cable-stayed spans."),
        "1L": (7900, None, "")}
WORDS = ("time year people way day man thing woman life child world school state family student group country "
         "problem hand part place case week company system program question work government number night point "
         "home water room mother area money story fact month lot right study book eye job word business issue "
         "side kind head").split()
CHUNK, BUDGET, RATIO, GRID = 512, 2048, 4, 256    # attention chunk, sparse budget, block ratio, score-grid tokens
POISON_SCORE = 1e30


def long_prompt(tok, tokens: int, ask: str = "") -> str:
    """Random-word prose of about ``tokens`` tokens, then ``ask`` (``LONG``: decodes that cross launch-bound steps)."""
    import random

    rng = random.Random(7)
    words: list[str] = []
    while True:
        words += [rng.choice(WORDS) for _ in range(200)]
        text = "Summarize the following notes in plain prose.\n\n" + " ".join(words) + (f"\n\n{ask}" if ask else "")
        if len(tok.encode(text).ids) >= tokens:
            return text


def prompts_of(app, name: str, prompts):
    return [long_prompt(app.tok, LONG[name][0], LONG[name][2])] if prompts is None else prompts


def count_of(name: str, count: int) -> int:
    return LONG[name][1] or count if name in LONG else count


FAILS: list[str] = []


def fail(msg: str) -> None:
    FAILS.append(msg)
    print(f"FAIL {msg}", flush=True)


def die(msg: str) -> None:
    """A FAIL inside an engine step: print it and end the process (the requests would wait on a dead engine)."""
    print(f"FAIL {msg}", flush=True)
    print("exit 1", flush=True)
    os._exit(1)


def body_of(prompt: str, count: int) -> dict:
    return {"model": "Qwen3.8-Flash-Next", "max_tokens": count, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "seed": 1234, "chat_template_kwargs": {"enable_thinking": False}}


def run(engine, app, prompts: list, count: int, stop_eos: bool = True) -> list[dict]:
    prepared = [app.prepare(body_of(p, count), True) for p in prompts]
    out: list = [None] * len(prompts)
    toks: list = [[] for _ in prompts]

    def worker(i: int) -> None:
        def feed(new, i=i):
            toks[i].extend(list(new))
        r = engine.generate(list(prepared[i].prompt), count, prepared[i].sampling, feed, draft=True, stop_eos=stop_eos)
        out[i] = {"sha": hashlib.sha256(json.dumps(toks[i]).encode()).hexdigest()[:16], "n": len(toks[i]),
                  "rounds": r.get("rounds"), "drafted": r.get("drafted"), "accepted": r.get("accepted")}

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(prompts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def as_int(t: torch.Tensor) -> torch.Tensor:
    """An integer view of ``t`` for a bit comparison (NaN payloads compared as bits)."""
    t = t.contiguous()
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def written(segs) -> list:
    """The state an MTP step of one row a stream writes: each stream's cache row at its MTP length (k, v, their
    scales, the indexer key) and, when that row completes a RATIO block, the pooled block."""
    out = []
    for st, _, _ in segs:
        p = st.mtp_len
        kc = st.mtp_kc
        out += [kc.k[p], kc.v[p], st.mtp_ikc[p]]
        if kc.quantized:
            out += [kc.ks[p], kc.vs[p]]
        if (p + 1) % RATIO == 0:
            out.append(st.mtp_pooled[p // RATIO])
    return out


def poison(segs, sc) -> None:
    """Overwrite what the eager step just wrote and the attention scratch a step fills: the replay must write it all
    again. Key lists stay valid ids (0) and key counts 1, so a missing launch changes bits without a wild read."""
    for t in written(segs):
        if not t.is_contiguous():
            die("a written-state view is not contiguous: its poison would land in a copy")
        as_int(t).fill_(0x5A if t.element_size() == 1 else 0x5A5A)
    sc.po.fill_(float("nan"))
    sc.pm.fill_(float("nan"))
    sc.pl.fill_(float("nan"))
    sc.ids.zero_()
    sc.nk.fill_(1)
    if sc.scores is not None:
        sc.scores.fill_(POISON_SCORE)


def crosses(c0: int, e: int) -> tuple[bool, bool]:
    """(a chunk step, a sparse score-grid step) between a graph's capture context ``c0`` and a replay's ``e``."""
    def cdiv(a, b):
        return -(-a // b)

    keys = (BUDGET // RATIO + 1) * RATIO - 1
    chunk = cdiv(min(e, keys), CHUNK) > cdiv(min(c0, keys), CHUNK)
    grid = e // RATIO > BUDGET // RATIO and cdiv(e, GRID) > cdiv(max(c0, BUDGET), GRID)
    return chunk, grid


def main() -> None:
    from tensorfold import hub
    from tensorfold.cuda.server import App
    from tensorfold.families.qwen4_exp import cuda_engine
    from tensorfold.families.qwen4_exp.cuda import attn_multi, multi
    from tensorfold.families.qwen4_exp.cuda.mtp import mtp_compute

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    model_dir = hub.resolve("Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP", download=False)
    engine = cuda_engine(model_dir, drafter="", tp=1, rank=0, master="", master_port=29551, no_drafts=False,
                         mtp_drafts=8, mtp_confidence=0.60, context=262144, context_explicit=True,
                         ple_on_ssd=True, kv_dtype="int8", parallel=5, vision=True, vision_urls=False)
    app = App(engine, model_dir, "Qwen3.8-Flash-Next", default_thinking=True,
              sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 20}, max_tokens=32768,
              context_window=262144, vision_max_images=50)
    md = engine.multi
    if md.chain_graphs is None:
        fail("chain graphs are off in the engine as built (TENSORFOLD_MTP_GRAPHS / no MTP head)")
    graphs = md.chain_graphs
    sc = md.mbuf.attn
    stats = {"replays": 0, "same": 0, "chunk": 0, "grid": 0, "ends": []}
    captured: dict = {}                                  # graph key -> the context it was captured at
    orig_run = multi.ChainGraphs.run

    def checked(self, segs):
        try:
            if not (all(a1 - a0 == 1 for _, a0, a1 in segs) and all(st.image_positions is None for st, _, _ in segs)):
                got = orig_run(self, segs)
                if got is not None:
                    die(f"a step of rows {[a1 - a0 for _, a0, a1 in segs]} ran as a graph")
                return got
            end = max(st.mtp_len + 1 for st, _, _ in segs)
            stats["ends"].append(end)
            key = self._key(segs, max(multi.GRAPH_BUCKET, 1 << (end - 1).bit_length()))
            if key not in self.graphs:
                got = orig_run(self, segs)
                if key in self.graphs:
                    captured[key] = end
                return got
            md.mbuf.attn_step = attn_multi.Step(md.w, segs, mtp=True)       # production's eager step
            try:
                ref = mtp_compute(md.w, segs, md.mbuf).clone()
            finally:
                md.mbuf.attn_step = None
            ref_state = [as_int(t).clone() for t in written(segs)]
            poison(segs, sc)
            got = orig_run(self, segs)
            torch.cuda.synchronize()
            stats["replays"] += 1
            if got is None:
                die(f"a captured key ran eagerly at context {end}")
            if not torch.equal(as_int(got), as_int(ref)):
                die(f"replay at context {end} (captured at {captured.get(key)}): draft logits differ from the "
                    "eager step's")
            for i, (a, t) in enumerate(zip(ref_state, written(segs))):
                if not torch.equal(a, as_int(t)):
                    die(f"replay at context {end} (captured at {captured.get(key)}): written state {i} differs "
                        f"from the eager step's (MTP lengths {[st.mtp_len for st, _, _ in segs]})")
            stats["same"] += 1
            if key in captured:
                chunk, grid = crosses(captured[key], end)
                stats["chunk"] += chunk
                stats["grid"] += grid
            return got
        except BaseException:
            traceback.print_exc()
            die("a graph step raised")

    for name, clients, prompts in TESTS:
        if ONLY and name not in ONLY:
            continue
        prompts = prompts_of(app, name, prompts)
        n = count_of(name, count)
        msgs = [f"{p}\n\n(test {name}-{i})" for i, p in enumerate(prompts[:clients])]
        eos = name not in LONG
        stats.update(replays=0, same=0, chunk=0, grid=0, ends=[])
        caps = graphs.captures
        multi.ChainGraphs.run = checked
        while md._evict_kept(None):                             # no kept prompt end: every run prefills alike
            pass
        on = run(engine, app, msgs, n, eos)
        multi.ChainGraphs.run = orig_run
        ends = stats["ends"] or [0]
        line = (f"{name}: {stats['replays']} replays checked against the eager step over poisoned state, "
                f"{stats['same']} identical; {graphs.captures - caps} captures; {stats['chunk']} replays past "
                f"their capture across a chunk step, {stats['grid']} across a score-grid step; one-row steps "
                f"at contexts {min(ends)}-{max(ends)}; {[o['n'] for o in on]} tokens")
        if stats["replays"] == 0 or stats["same"] != stats["replays"]:
            fail(line)
        elif name == "1M" and not (stats["chunk"] and stats["grid"]):
            fail(line + " (1M must cross both inside one graph)")
        else:
            print(f"PASS {line}", flush=True)
        md.chain_graphs = None                                   # the same requests eagerly
        while md._evict_kept(None):
            pass
        off = run(engine, app, msgs, n, eos)
        md.chain_graphs = graphs                                 # and on, unchecked: production's path
        while md._evict_kept(None):
            pass
        pure = run(engine, app, msgs, n, eos)
        for i, (a, b, c) in enumerate(zip(on, off, pure)):
            line = (f"{name} stream {i}: checked {a['sha']} {a['n']} tok {a['rounds']} rounds {a['accepted']} "
                    f"accepted, eager {b['sha']} {b['rounds']} rounds {b['accepted']} accepted, graphs "
                    f"{c['sha']} {c['rounds']} rounds {c['accepted']} accepted")
            if not a == b == c:
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
