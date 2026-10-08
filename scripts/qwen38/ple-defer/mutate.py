#!/usr/bin/env python3
"""mutate.py <mutant|--list> <layer dir> <out dir>: write ple-defer's forward.py and multi.py from <layer dir> into
<out dir> with one mutant applied (every anchor must match EXACTLY once: exit 2 otherwise). Kept in Python, not in
shell strings, so no quoting layer can corrupt an anchor."""
from __future__ import annotations

import pathlib
import sys

CUDA = "tensorfold/families/qwen4_exp/cuda"
DEFER = "defer_ple=PLE_DEFER and not (self.filling and self.converged)"
CALL = "        staging()\n"
TAIL = "c.eps, c.streams, c.ngram_size)\n\n\ndef stage_ple_rows"
MUTANTS = {   # name -> [(file, old, new)]
    "defer-never": [("multi.py", DEFER, "defer_ple=False")],
    "staging-dropped": [("forward.py", CALL, "")],
    "staging-late": [("forward.py", CALL, ""),
                     ("forward.py", TAIL, "c.eps, c.streams, c.ngram_size)\n    if staging is not None:\n"
                                          "        staging()\n\n\ndef stage_ple_rows")],
}


def main() -> None:
    if sys.argv[1] == "--list":
        print(" ".join(MUTANTS))
        return
    name, src, out = sys.argv[1], pathlib.Path(sys.argv[2]) / CUDA, pathlib.Path(sys.argv[3])
    files = {f: (src / f).read_text() for f in ("forward.py", "multi.py")}
    for f, old, new in MUTANTS[name]:
        n = files[f].count(old)
        if n != 1:
            print(f"{name}: anchor in {f} matched {n} times: {old[:60]!r}")
            sys.exit(2)
        files[f] = files[f].replace(old, new)
    out.mkdir(parents=True, exist_ok=True)
    for f, text in files.items():
        (out / f).write_text(text)


if __name__ == "__main__":
    main()
