#!/usr/bin/env python3
"""stock_sum.py stockab.jsonl: stock_ab.sh's table. Decode rows carry label <arm>-p<pass>; qf_run.py labels its
prefill rows prefill-<tokens> without the arm, so each is given to the arm-pass whose decode rows came just before
it (qf_run.py writes decode then prefill for one label). Per (kind, clients): per-request decode tok/s, the median
of each pass's median and the passes' spread; prefill tok/s per size; and whether every reply is byte-identical
across arms and passes."""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict


def main() -> None:
    rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
    dec = defaultdict(lambda: defaultdict(list))      # (kind, clients) -> (arm, pass) -> [tok/s]
    pre = defaultdict(lambda: defaultdict(list))      # tokens -> arm -> [tok/s]
    shas = defaultdict(set)                           # (kind, clients, prompt) -> {sha}
    current = None
    for r in rows:
        label = r["label"]
        if label.startswith("prefill-"):
            if current is None:
                sys.exit("a prefill row before any decode row: cannot attribute it")
            pre[int(label.split("-", 1)[1])][current[0]].append(r["prompt_tokens"] / r["prefill_s"])
            continue
        arm, p = label.rsplit("-p", 1)
        current = (arm, int(p))
        if r.get("decode_s"):
            dec[(r["kind"], r["clients"])][current].append(r["completion_tokens"] / r["decode_s"])
            shas[(r["kind"], r["clients"], r["prompt"])].add(r["sha"])
    arms = sorted({a for cell in dec.values() for a, _ in cell})
    print("decode, per-request tok/s: median of the passes' medians [spread of the pass medians]")
    for key in sorted(dec):
        out = []
        for arm in arms:
            meds = [statistics.median(v) for (a, _), v in sorted(dec[key].items()) if a == arm]
            out.append(f"{arm} {statistics.median(meds):6.1f} [{min(meds):.1f}-{max(meds):.1f}]")
        print(f"  {key[0]:6s} C={key[1]}: " + "   ".join(out))
    print("prefill tok/s: median of the passes [range]")
    for n in sorted(pre):
        print(f"  {n:7d}: " + "   ".join(f"{a} {statistics.median(v):7.0f} [{min(v):.0f}-{max(v):.0f}]"
                                         for a, v in sorted(pre[n].items())))
    diff = [k for k, s in shas.items() if len(s) != 1]
    print(f"replies: {len(shas)} prompts, {'every reply byte-identical' if not diff else f'{len(diff)} DIFFER'}")


if __name__ == "__main__":
    main()
