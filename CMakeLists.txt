cmake_minimum_required(VERSION 3.24)
project(numcore VERSION 0.1.0 LANGUAGES CXX)

# ---- Options ----------------------------------------------------------------
option(NUMCORE_BUILD_TESTS "Build the unit tests" ON)
option(NUMCORE_BUILD_BENCHMARKS "Build the Google Benchmark programs" ON)
# Off by default on purpose: P1 measures what these flags do.
# -march=native = "use every instruction THIS CPU has" (AVX2 + FMA on the
# i7-12700H), instead of the safe default that runs on any x86-64 CPU (SSE2).
# Binaries built with it may crash on older CPUs, so it's off by default.
# Use the release-native preset (build-native/) for P1 version 4.
option(NUMCORE_NATIVE "Compile with -O3 -march=native" OFF)

# ---- Global settings ---------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Library -----------------------------------------------------------------
add_library(numcore
  src/version.cpp
  src/matmul.cpp
        include/numcore/matmul.hpp
)
add_library(numcore::numcore ALIAS numcore)

target_include_directories(numcore PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_options(numcore PRIVATE -Wall -Wextra -Wpedantic)

# OpenMP (P1 version 5): a compiler feature for running loops on several
# threads. find_package checks the compiler supports it; linking the
# OpenMP::OpenMP_CXX target adds -fopenmp when compiling and the OpenMP
# runtime library (libgomp for GCC) when linking. Without -fopenmp, the
# `#pragma omp` lines are silently ignored and everything runs on 1 thread.
find_package(OpenMP REQUIRED)
target_link_libraries(numcore PRIVATE OpenMP::OpenMP_CXX)

# Eigen (P1 version 6): a widely used C++ linear-algebra library, used here
# as the "professional" matrix multiply to compare against. Header-only: there
# is nothing to compile or link, just headers to #include.
# Uses the system copy if installed (Ubuntu: sudo apt install libeigen3-dev),
# otherwise downloads it. Linked PRIVATE: only src/matmul.cpp sees Eigen; the
# public header numcore/matmul.hpp doesn't mention it.
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
  message(STATUS "numcore: Eigen 3.4 not found on the system, downloading it")
  include(FetchContent)
  FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    # Only download; don't run Eigen's own (large) CMake project.
    SOURCE_SUBDIR do-not-build
  )
  FetchContent_MakeAvailable(eigen)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}")
endif()
target_link_libraries(numcore PRIVATE Eigen3::Eigen)

if(NUMCORE_NATIVE)
  target_compile_options(numcore PUBLIC -O3 -march=native)
endif()

# ---- Tests -------------------------------------------------------------------
if(NUMCORE_BUILD_TESTS)
  enable_testing()

  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_subdirectory(tests)
endif()

# ---- Benchmarks --------------------------------------------------------------
# Benchmarks are only built in Release. A Debug build has no optimisation
# (-O0): every C(i, j) is a real function call and nothing goes in registers,
# so timings measure the missing optimiser, not the algorithm. Skipping the
# target entirely means there's nothing to run by mistake (e.g. from CLion's
# Debug profile), and Debug builds don't download Google Benchmark either.
if(NUMCORE_BUILD_BENCHMARKS AND NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "numcore: benchmarks skipped (build type is "
                 "'${CMAKE_BUILD_TYPE}', benchmarks need Release)")
endif()

if(NUMCORE_BUILD_BENCHMARKS AND CMAKE_BUILD_TYPE STREQUAL "Release")
  include(FetchContent)
  FetchContent_Declare(
    googlebenchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  # Only build the library: skip Google Benchmark's own tests and install rules.
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googlebenchmark)

  add_subdirectory(bench)
endif()
