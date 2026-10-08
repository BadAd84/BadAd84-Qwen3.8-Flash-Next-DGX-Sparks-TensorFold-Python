#!/usr/bin/env python3
"""mutate.py <mutant|--list> <layer dir> <out dir>: write hc-split's qmm.py from <layer dir> into <out dir> with one
mutant applied (every anchor must match EXACTLY once: exit 2 otherwise). Kept in Python, not in shell strings, so no
quoting layer can corrupt an anchor."""
from __future__ import annotations

import pathlib
import sys

QMM = "tensorfold/families/qwen4_exp/cuda/qmm.py"
SUM = "    total += v0\n    total += v1\n    total += v2\n    total += v3\n"
MUTANTS = {   # name -> [(old, new)]
    "sum-as-tree": [(SUM, "    total += (v0 + v1) + (v2 + v3)\n")],
    "sum-order": [(SUM, "    total += v0\n    total += v2\n    total += v1\n    total += v3\n")],
    "sum-from-v0": [("    total = tl.zeros((BM, DB), dtype=tl.float32)\n    total += v0\n", "    total = v0\n")],
    "stream-major-columns": [("    rn = (n % 4) * D + j * DB + n // 4", "    rn = (n // DB) * D + j * DB + n % DB")],
    "wide-off": [('HC_WIDE = os.environ.get("TENSORFOLD_HC_WIDE", "1") != "0"', "HC_WIDE = False")],
    "group-sum-tl": [("    tl.store(XSM + rm * (D // 32) + j, r0 + r1, mask=m_ok)",
                      "    tl.store(XSM + rm * (D // 32) + j, tl.sum(mixed.to(tl.float32), axis=1), mask=m_ok)")],
    "group-run-tree": [("    r = ((e0 + e1) + e2) + e3", "    r = (e0 + e1) + (e2 + e3)")],
}


def main() -> None:
    if sys.argv[1] == "--list":
        print(" ".join(MUTANTS))
        return
    name, src, out = sys.argv[1], pathlib.Path(sys.argv[2]) / QMM, pathlib.Path(sys.argv[3])
    text = src.read_text()
    for old, new in MUTANTS[name]:
        n = text.count(old)
        if n != 1:
            print(f"{name}: anchor matched {n} times: {old[:60]!r}")
            sys.exit(2)
        text = text.replace(old, new)
    out.mkdir(parents=True, exist_ok=True)
    (out / "qmm.py").write_text(text)


if __name__ == "__main__":
    main()
