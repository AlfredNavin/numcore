#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# P1 final benchmark run: all 6 matrix-multiply versions at 1024 and 2048.
#
#   ./scripts/run_final_bench.sh            # full run (~10-15 min)
#   REPS=1 ./scripts/run_final_bench.sh     # quick check that everything works
#
# Before the real run (for numbers you publish):
#   1. Plug in the charger (on battery the CPU slows itself down).
#   2. Close the browser, CLion, etc.
#   3. Performance governor (stops the CPU dropping its clock speed between runs):
#        echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
#      and back afterwards:
#        echo powersave | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
#
# What it runs (3 passes, each saved as JSON in results/):
#   p1_release.json        build/        (Release, no -march=native), pinned to CPU 4
#                          -> versions 1-3: naive, i-k-j, tiled 128
#   p1_native_1thread.json build-native/ (Release + -march=native), pinned to CPU 4
#                          -> version 4: tiled 128 with AVX2/FMA; Eigen on 1 thread
#   p1_native_threads.json build-native/, NOT pinned (threads need many CPUs)
#                          -> version 5 and Eigen at 1, 2, 4, 6, 8, 12, 20 threads
# Then scripts/plot_p1.py turns them into charts + a summary table.
# ----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."   # run from the repo root, wherever it's called from

REPS="${REPS:-5}"          # repetitions per benchmark; the median is used
PIN_CPU="${PIN_CPU:-4}"    # a fast P-core (4.7 GHz); its hyperthread twin is CPU 5
OUT="${OUT:-results}"   # where the JSON, charts and table go
mkdir -p "$OUT"

# ---- 1. Check the machine is ready (warn, don't stop) --------------------------
governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)
if [[ "$governor" != "performance" ]]; then
  echo "WARNING: CPU governor is '$governor', not 'performance'. Numbers will be noisier."
  echo "         See the top of this script for how to switch."
fi
on_ac=$(cat /sys/class/power_supply/A{C,DP}*/online 2>/dev/null | head -n1 || true)
if [[ "$on_ac" == "0" ]]; then
  echo "WARNING: running on battery. Plug in the charger for the real run."
fi

# ---- 2. Build both configurations and run the tests ----------------------------
# Never benchmark code that fails its tests: a fast wrong answer is worthless.
for preset in release release-native; do
  dir=build; [[ "$preset" == release-native ]] && dir=build-native
  cmake --preset "$preset" > /dev/null
  cmake --build "$dir"
  ctest --preset "$preset" > /dev/null || { echo "Tests failed for $preset: fix them first."; exit 1; }
done
echo "Builds OK, tests pass."

# ---- 3. Record how the run was done (goes in the write-up) ----------------------
{
  echo "date:     $(date --iso-8601=seconds)"
  echo "commit:   $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (+ uncommitted changes)')"
  echo "cpu:      $(lscpu | sed -n 's/^Model name: *//p')"
  echo "governor: $governor"
  echo "on AC:    ${on_ac:-unknown}"
  echo "reps:     $REPS (median reported)"
  echo "pinned:   single-thread runs on CPU $PIN_CPU; multithreaded runs unpinned"
  echo "compiler: $(c++ --version | head -n1)"
} > "$OUT/p1_run_info.txt"
cat "$OUT/p1_run_info.txt"

# Flags shared by every pass:
#   repetitions + aggregates_only -> report mean/median/stddev/cv of REPS runs
common=(--benchmark_repetitions="$REPS"
        --benchmark_report_aggregates_only=true
        --benchmark_out_format=json)

# ---- 4. The three passes ----------------------------------------------------------
echo; echo "== Pass 1/3: versions 1-3 (build/, CPU $PIN_CPU). Naive 2048 is slow, be patient."
taskset -c "$PIN_CPU" ./build/bench/numcore_bench "${common[@]}" \
  --benchmark_filter='^BM_(Naive|Ikj)/(1024|2048)$|^BM_Tiled/n:(1024|2048)/tile:128$' \
  --benchmark_out="$OUT/p1_release.json"

echo; echo "== Pass 2/3: version 4 + Eigen 1 thread (build-native/, CPU $PIN_CPU)"
taskset -c "$PIN_CPU" ./build-native/bench/numcore_bench "${common[@]}" \
  --benchmark_filter='^BM_Tiled/n:(1024|2048)/tile:128$|^BM_Eigen/n:(1024|2048)/threads:1/' \
  --benchmark_out="$OUT/p1_native_1thread.json"

echo; echo "== Pass 3/3: version 5 + Eigen, thread sweep (build-native/, unpinned)"
./build-native/bench/numcore_bench "${common[@]}" \
  --benchmark_filter='^BM_(Parallel|Eigen)/n:(1024|2048)/' \
  --benchmark_out="$OUT/p1_native_threads.json"

# ---- 5. Charts + summary table -----------------------------------------------------
echo
PY=python3
[[ -x .venv/bin/python ]] && PY=.venv/bin/python
"$PY" scripts/plot_p1.py "$OUT"
