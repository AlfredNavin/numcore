add_executable(numcore_bench
  bench_matmul.cpp
)
target_link_libraries(numcore_bench PRIVATE numcore::numcore benchmark::benchmark_main)
target_compile_options(numcore_bench PRIVATE -Wall -Wextra -Wpedantic)

# Timing a Debug build measures the missing optimiser, not your algorithm.
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(WARNING "numcore_bench is being built as '${CMAKE_BUILD_TYPE}'. "
                  "Use the release preset for numbers you report.")
endif()
