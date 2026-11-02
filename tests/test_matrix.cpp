#include "numcore/matrix.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

using numcore::Matrix;

// TEST(GroupName, TestName). EXPECT_* records a failure and keeps going;
// ASSERT_* stops this test immediately (use it when continuing would crash).

// ---- Construction -------------------------------------------------------------

TEST(MatrixConstruct, DefaultIsEmpty) {
  Matrix m;
  EXPECT_EQ(m.rows(), 0u);
  EXPECT_EQ(m.cols(), 0u);
  EXPECT_EQ(m.size(), 0u);
}

TEST(MatrixConstruct, SizedIsZeroFilled) {
  Matrix m(2, 3);
  ASSERT_EQ(m.rows(), 2u);
  ASSERT_EQ(m.cols(), 3u);
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 3; ++j) EXPECT_EQ(m(i, j), 0.0);
}

TEST(MatrixConstruct, FillValue) {
  Matrix m(3, 2, 7.5);
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 2; ++j) EXPECT_EQ(m(i, j), 7.5);
}

TEST(MatrixConstruct, InitializerList) {
  Matrix m{{1, 2, 3},
           {4, 5, 6}};
  ASSERT_EQ(m.rows(), 2u);
  ASSERT_EQ(m.cols(), 3u);
  EXPECT_EQ(m(0, 0), 1.0);
  EXPECT_EQ(m(0, 2), 3.0);
  EXPECT_EQ(m(1, 0), 4.0);
  EXPECT_EQ(m(1, 2), 6.0);
}

TEST(MatrixConstruct, RaggedInitializerListThrows) {
  EXPECT_THROW((Matrix{{1, 2}, {3}}), std::invalid_argument);
}

TEST(MatrixConstruct, Identity) {
  Matrix I = Matrix::identity(3);
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) EXPECT_EQ(I(i, j), i == j ? 1.0 : 0.0);
}

// ---- Layout and access ----------------------------------------------------------

TEST(MatrixAccess, IsRowMajor) {
  Matrix m{{1, 2, 3},
           {4, 5, 6}};
  const double* p = m.data();
  for (int k = 0; k < 6; ++k) EXPECT_EQ(p[k], k + 1.0);  // 1 2 3 4 5 6
}

TEST(MatrixAccess, WriteThenRead) {
  Matrix m(2, 2);
  m(1, 0) = 42.0;
  EXPECT_EQ(m(1, 0), 42.0);
  EXPECT_EQ(m(0, 0), 0.0);  // neighbours untouched
}

TEST(MatrixAccess, AtThrowsOutOfRange) {
  Matrix m(2, 3);
  EXPECT_NO_THROW(m.at(1, 2));
  EXPECT_THROW(m.at(2, 0), std::out_of_range);
  EXPECT_THROW(m.at(0, 3), std::out_of_range);
}

// ---- Arithmetic -----------------------------------------------------------------

TEST(MatrixAdd, ElementWise) {
  Matrix a{{1, 2}, {3, 4}};
  Matrix b{{10, 20}, {30, 40}};
  Matrix expected{{11, 22}, {33, 44}};
  EXPECT_EQ(a + b, expected);
  EXPECT_EQ(a(0, 0), 1.0);  // operator+ must not modify its inputs
}

TEST(MatrixAdd, PlusEqualsModifiesInPlace) {
  Matrix a{{1, 2}, {3, 4}};
  a += Matrix(2, 2, 1.0);
  EXPECT_EQ(a, (Matrix{{2, 3}, {4, 5}}));
}

TEST(MatrixAdd, ShapeMismatchThrows) {
  Matrix a(2, 3);
  Matrix b(3, 2);
  EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(MatrixTranspose, NonSquare) {
  Matrix m{{1, 2, 3},
           {4, 5, 6}};
  Matrix expected{{1, 4},
                  {2, 5},
                  {3, 6}};
  EXPECT_EQ(numcore::transpose(m), expected);
}

TEST(MatrixTranspose, TwiceIsOriginal) {
  Matrix m{{1, 2, 3}, {4, 5, 6}};
  EXPECT_EQ(numcore::transpose(numcore::transpose(m)), m);
}

// ---- Approximate equality -------------------------------------------------------

TEST(MatrixApproxEqual, ToleratesRoundingError) {
  Matrix a{{0.1 + 0.2}};
  Matrix b{{0.3}};
  EXPECT_FALSE(a == b);  // 0.1 + 0.2 is 0.30000000000000004
  EXPECT_TRUE(numcore::approx_equal(a, b));
}

TEST(MatrixApproxEqual, DetectsRealDifference) {
  Matrix a{{1.0, 2.0}};
  Matrix b{{1.0, 2.001}};
  EXPECT_FALSE(numcore::approx_equal(a, b));
}

TEST(MatrixApproxEqual, ShapeMismatchIsNotEqual) {
  EXPECT_FALSE(numcore::approx_equal(Matrix(2, 3), Matrix(3, 2)));
}

TEST(MatrixApproxEqual, NaNIsNeverEqual) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(numcore::approx_equal(Matrix{{nan}}, Matrix{{nan}}));
}
