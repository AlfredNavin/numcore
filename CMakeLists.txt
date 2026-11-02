cmake_minimum_required(VERSION 3.24)
project(numcore VERSION 0.1.0 LANGUAGES CXX)

# ---- Options ----------------------------------------------------------------
option(NUMCORE_BUILD_TESTS "Build the unit tests" ON)
# Off by default on purpose: P1 measures what these flags do.
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
)
add_library(numcore::numcore ALIAS numcore)

target_include_directories(numcore PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_options(numcore PRIVATE -Wall -Wextra -Wpedantic)

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
