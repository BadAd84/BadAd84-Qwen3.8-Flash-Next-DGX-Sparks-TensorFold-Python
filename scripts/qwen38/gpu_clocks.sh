#!/usr/bin/env bash
# gpu_clocks.sh   on a Spark in ~/work: what the box does under gpu_equiv.py's load, to explain a matmul difference
# between two boxes. The top CPU users at idle first (a resident server busy at idle shares the GB10's power budget
# with the GPU), then the SM clock, power and temperature sampled every 0.5 s while gpu_equiv.py runs (median and min).
set -uo pipefail
cd "$HOME/work" || exit 1
echo "== $(hostname) $(date +%T): top CPU at idle"
top -bn1 -o %CPU | sed -n '7,11p' | cut -c1-110
img=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -m1 'tensorfold-qwen38:zig')
nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits -lms 500 > clocks.csv &
smi=$!
docker run --rm --gpus all -v "$HOME/work":/w --entrypoint python3 "$img" /w/gpu_equiv.py 2>&1 | grep -E "GB/s|TFLOPS"
kill "$smi"
python3 - <<'PY'
import statistics
rows = []
for line in open("clocks.csv"):
    try:
        mhz, watts, temp = (float(x) for x in line.split(","))
        rows.append((mhz, watts, temp))
    except ValueError:
        pass
busy = [r for r in rows if r[1] > 20]
mhz, watts, temp = zip(*busy)
print(f"under load ({len(busy)} samples): SM clock median {statistics.median(mhz):.0f} min {min(mhz):.0f} MHz, "
      f"power median {statistics.median(watts):.1f} max {max(watts):.1f} W, temp max {max(temp):.0f} C")
PY
rm -f clocks.csv
