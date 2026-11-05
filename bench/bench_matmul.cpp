// ----------------------------------------------------------------------------
// Benchmarks for the P1 matrix-multiply versions.
//
// How Google Benchmark works:
//   - Each BENCHMARK(...) registers a function that takes a benchmark::State.
//   - `for (auto _ : state) { ... }` is the timed loop. The library decides
//     how many times to run it until the timing is stable, then reports the
//     average time per iteration.
//   - Anything BEFORE the loop (building inputs) is setup and is NOT timed.
//   - ->Arg(n) runs the same benchmark once per size; state.range(0) reads n.
//
// Run (always from a Release build):
//   ./build/bench/numcore_bench
//   ./build/bench/numcore_bench --benchmark_filter=Naive/512
//   ./build/bench/numcore_bench --benchmark_repetitions=5
//       --benchmark_out=results/matmul.json --benchmark_out_format=json
//   (that's one command, split here for width)
// ----------------------------------------------------------------------------
#include "numcore/matmul.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <random>

using numcore::Matrix;

namespace {

// n x n matrix of random values in [-1, 1). Fixed seed = same inputs every run,
// so runs are comparable. Random (not all-ones) so the compiler can't cheat.
Matrix random_matrix(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  Matrix m(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) m(i, j) = dist(rng);
  return m;
}

// Shared body for every multiply version. `Multiply` is any function with the
// signature Matrix(const Matrix&, const Matrix&): pass multiply_naive,
// multiply_ikj, ...
template <typename Multiply>
void run_matmul(benchmark::State& state, Multiply multiply) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const Matrix A = random_matrix(n, 1);
  const Matrix B = random_matrix(n, 2);

  for (auto _ : state) {
    Matrix C = multiply(A, B);
    // Tell the compiler C is "used", so it can't delete the whole multiply
    // as dead code.
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  // An n x n multiply does n^3 multiplies + n^3 adds = 2n^3 floating-point
  // operations. kIsIterationInvariantRate = "this many per iteration, report
  // it per second", so the output column FLOPS is your GFLOPS number
  // (e.g. 1.5G/s = 1.5 GFLOPS).
  const double flops = 2.0 * static_cast<double>(n) * static_cast<double>(n) *
                       static_cast<double>(n);
  state.counters["FLOPS"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
}

// ---- Version 1: naive i-j-k ---------------------------------------------------
void BM_Naive(benchmark::State& state) { run_matmul(state, numcore::multiply_naive); }

// Sizes from the plan: 256/512/1024 now, 2048 for the final chart.
// (2048 naive can take a minute per run; add ->Arg(2048) when you're ready.)
BENCHMARK(BM_Naive)->Arg(256)->Arg(512)->Arg(1024)->Unit(benchmark::kMillisecond);

// ---- Version 2: i-k-j -----------------------------------------------------------
void BM_Ikj(benchmark::State& state) { run_matmul(state, numcore::multiply_ikj); }
BENCHMARK(BM_Ikj)->Arg(256)->Arg(512)->Arg(1024)->Arg(2048)->Unit(benchmark::kMillisecond);

// ---- Version 3: cache blocking (tiling) ---------------------------------------
// Two arguments per run: matrix size n and tile size T, so you can see which
// tile size suits this CPU's caches. ArgsProduct runs every combination.
// (multiply_tiled takes a 3rd parameter, so it's wrapped in a lambda that
// fixes the tile size; run_matmul only ever passes A and B.)
void BM_Tiled(benchmark::State& state) {
  const auto tile = static_cast<std::size_t>(state.range(1));
  run_matmul(state, [tile](const Matrix& A, const Matrix& B) {
    return numcore::multiply_tiled(A, B, tile);
  });
}
BENCHMARK(BM_Tiled)
    ->ArgsProduct({{256, 512, 1024, 2048}, {16, 32, 64, 128, 256}})
    ->ArgNames({"n", "tile"})
    ->Unit(benchmark::kMillisecond);

}  // namespace

// No main(): linking benchmark::benchmark_main provides one.
