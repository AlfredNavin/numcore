add_executable(numcore_tests
  test_matrix.cpp
  test_matmul.cpp
        ../include/numcore/matmul.hpp
)
target_link_libraries(numcore_tests PRIVATE numcore::numcore GTest::gtest_main)
target_compile_options(numcore_tests PRIVATE -Wall -Wextra -Wpedantic)

include(GoogleTest)
gtest_discover_tests(numcore_tests)
