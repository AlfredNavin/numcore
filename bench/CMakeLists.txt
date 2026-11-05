add_executable(numcore_bench
  bench_matmul.cpp
)
target_link_libraries(numcore_bench PRIVATE numcore::numcore benchmark::benchmark_main)
target_compile_options(numcore_bench PRIVATE -Wall -Wextra -Wpedantic)

# Only reached in Release builds: see the Benchmarks section of the root
# CMakeLists.txt.
