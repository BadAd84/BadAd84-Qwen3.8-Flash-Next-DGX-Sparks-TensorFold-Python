#!/usr/bin/env python3
"""prefill_fresh.py <arm> <out.jsonl> [seed_base=700000]: cold prefill at 8k / 32k / 128k tokens, 3 runs each, every
prompt new to the server. qf_run.py's prefill prompts repeat on every pass (seed 900000 + tokens + run * 7919 with
RUNS=1), so a server that keeps prompts answers passes 2-3 from its cache (the Zig release did: a 128k prompt in
0.31 s); this uses another seed base, one prompt per (size, run). Client-side rate = prompt tokens / time to the first
token. Prints each request and the median per size."""
import json
import statistics
import sys

import qf_harness as H


def main():
    arm, out = sys.argv[1], sys.argv[2]
    base = int(sys.argv[3]) if len(sys.argv) > 3 else 700000
    rates = {}
    with open(out, "a", encoding="utf-8") as f:
        for tokens in (8000, 32000, 128000):
            for run in range(3):
                r = H.complete(H.prose(tokens, base + tokens + run * 7919), 16)
                rate = r["prompt_tokens"] / r["ttft"]
                rates.setdefault(tokens, []).append(rate)
                f.write(json.dumps({"arm": arm, "tokens": tokens, "run": run, "prompt_tokens": r["prompt_tokens"],
                                    "ttft": r["ttft"], "rate": rate}) + "\n")
                print(f"{arm} prefill {tokens} run{run}: {r['prompt_tokens']} tok, TTFT {r['ttft']:.2f} s, "
                      f"{rate:.0f} tok/s", flush=True)
    for tokens, v in rates.items():
        print(f"{arm} prefill {tokens}: median {statistics.median(v):.0f} tok/s [{min(v):.0f}-{max(v):.0f}]")


if __name__ == "__main__":
    main()
