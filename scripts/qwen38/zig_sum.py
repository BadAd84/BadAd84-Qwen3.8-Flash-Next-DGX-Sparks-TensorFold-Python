#!/usr/bin/env python3
"""zig_sum.py zigab.jsonl: zig_ab.sh's table, the Qwen recipe's Zig release against our production. The two arms run
different engines and checkpoints, so the replies are not compared byte for byte, and the speeds are measured the same
way for both from the client's side (the Zig server may not send TensorFold's per-request stats block): decode tok/s
of a request = (completion tokens - 1) / (elapsed - time to the first token), prefill tok/s = prompt tokens / time to
the first token. Labels as stock_sum.py: decode rows <arm>-p<pass>, prefill rows prefill-<tokens> after their arm's
decode rows. Per (kind, clients): the median of each pass's median and the passes' spread; prefill per size."""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict


def decode_rate(r: dict) -> float | None:
    n, el, ft = r.get("completion_tokens"), r.get("elapsed"), r.get("ttft")
    if not n or n < 2 or el is None or ft is None or el <= ft:
        return None
    return (n - 1) / (el - ft)


def main() -> None:
    rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
    dec = defaultdict(lambda: defaultdict(list))      # (kind, clients) -> (arm, pass) -> [tok/s]
    pre = defaultdict(lambda: defaultdict(list))      # tokens -> arm -> [tok/s]
    toks = defaultdict(lambda: defaultdict(list))     # (kind, clients) -> arm -> [completion tokens]
    current = None
    for r in rows:
        label = r["label"]
        if label.startswith("prefill-"):
            if current is None:
                sys.exit("a prefill row before any decode row: cannot attribute it")
            if r.get("ttft") and r.get("prompt_tokens"):
                pre[int(label.split("-", 1)[1])][current[0]].append(r["prompt_tokens"] / r["ttft"])
            continue
        arm, p = label.rsplit("-p", 1)
        current = (arm, int(p))
        rate = decode_rate(r)
        if rate is not None:
            dec[(r["kind"], r["clients"])][current].append(rate)
            toks[(r["kind"], r["clients"])][arm].append(r["completion_tokens"])
    arms = sorted({a for cell in dec.values() for a, _ in cell})
    print("decode, per-request tok/s (client side): median of the passes' medians [spread of the pass medians]")
    for key in sorted(dec):
        out = []
        for arm in arms:
            meds = [statistics.median(v) for (a, _), v in sorted(dec[key].items()) if a == arm]
            if meds:
                out.append(f"{arm} {statistics.median(meds):6.1f} [{min(meds):.1f}-{max(meds):.1f}]")
        lens = "  reply tokens " + " / ".join(f"{a} {statistics.median(toks[key][a]):.0f}" for a in arms
                                               if toks[key][a])
        print(f"  {key[0]:6s} C={key[1]}: " + "   ".join(out) + lens)
    print("prefill tok/s (prompt tokens / time to the first token): median of the passes [range]")
    for n in sorted(pre):
        print(f"  {n:7d}: " + "   ".join(f"{a} {statistics.median(v):7.0f} [{min(v):.0f}-{max(v):.0f}]"
                                         for a, v in sorted(pre[n].items())))


if __name__ == "__main__":
    main()
