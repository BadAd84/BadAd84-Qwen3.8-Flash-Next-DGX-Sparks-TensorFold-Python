#!/usr/bin/env python3
"""gpu_equiv.py: the same short GPU measurement on two Sparks, so a two-box A/B can say how alike the boxes are.
Device memory copy bandwidth (2 GiB fp32 buffers, read + write counted) and bf16 matmul throughput (8192^3), 5 timed
repeats of 10 each with CUDA events after 3 warm-ups; prints every repeat and the median. Run in any image with
torch on the box idle: docker run --rm --gpus all -v ~/work:/w --entrypoint python3 <image> /w/gpu_equiv.py"""
import statistics

import torch


def timed(fn, reps=5, inner=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / inner / 1e3)
    return out


def main():
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}, {props.multi_processor_count} SMs, torch {torch.__version__}, cuda {torch.version.cuda}")
    n = 512 * 2 ** 20
    a = torch.ones(n, dtype=torch.float32, device=dev)
    b = torch.empty_like(a)
    copy = [2 * n * 4 / s / 1e9 for s in timed(lambda: b.copy_(a))]
    print("copy GB/s", [round(x, 1) for x in copy], "median", round(statistics.median(copy), 1))
    m = 8192
    x = torch.randn(m, m, dtype=torch.bfloat16, device=dev)
    y = torch.randn(m, m, dtype=torch.bfloat16, device=dev)
    mm = [2 * m ** 3 / s / 1e12 for s in timed(lambda: x @ y)]
    print("bf16 matmul TFLOPS", [round(t, 1) for t in mm], "median", round(statistics.median(mm), 1))


if __name__ == "__main__":
    main()
