#include "numcore/matmul.hpp"
#include "matrix_printer.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

using numcore::Matrix;
using numcore::multiply_naive;
using numcore::transpose;

// Pattern for every test:
//   1. Arrange: build the inputs and the expected answer (worked out by hand)
//   2. Act:     call the function
//   3. Assert:  compare actual vs expected
//
// All inputs are small integers, so every product and sum is exact in double
// and plain == is safe. (Real-valued inputs would need approx_equal.)

namespace {

// Deterministic rows x cols matrix of small integers in [-5, 5].
// Different `seed`s give different matrices; no randomness, so failures
// are reproducible.
Matrix make_int_matrix(std::size_t rows, std::size_t cols, std::size_t seed) {
  Matrix m(rows, cols);
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < cols; ++j)
      m(i, j) = static_cast<double>((i * 7 + j * 3 + seed * 5) % 11) - 5.0;
  return m;
}

}  // namespace

// ---- Hand-checked values -------------------------------------------------------

TEST(MultiplyNaive, HandChecked2x2) {
  Matrix A{{1, 2},
           {3, 4}};
  Matrix B{{5, 6},
           {7, 8}};
  // C(0,0) = 1*5 + 2*7 = 19    C(0,1) = 1*6 + 2*8 = 22
  // C(1,0) = 3*5 + 4*7 = 43    C(1,1) = 3*6 + 4*8 = 50
  Matrix expected{{19, 22},
                  {43, 50}};

  Matrix C = multiply_naive(A, B);

  EXPECT_EQ(C, expected);
}

TEST(MultiplyNaive, NonSquare) {
  // (2x3) * (3x2) -> 2x2
  Matrix A{{1, 2, 3},
           {4, 5, 6}};
  Matrix B{{7, 8},
           {9, 10},
           {11, 12}};
  // C(0,0) = 1*7 + 2*9 + 3*11  = 58     C(0,1) = 1*8 + 2*10 + 3*12 = 64
  // C(1,0) = 4*7 + 5*9 + 6*11  = 139    C(1,1) = 4*8 + 5*10 + 6*12 = 154
  Matrix expected{{58, 64},
                  {139, 154}};

  Matrix C = multiply_naive(A, B);

  ASSERT_EQ(C.rows(), 2u);
  ASSERT_EQ(C.cols(), 2u);
  EXPECT_EQ(C, expected);
}

TEST(MultiplyNaive, NotCommutative) {
  // Swapping the operands changes the answer: A*B != B*A in general.
  Matrix A{{1, 2},
           {3, 4}};
  Matrix B{{5, 6},
           {7, 8}};
  Matrix expected_BA{{23, 34},
                     {31, 46}};

  EXPECT_EQ(multiply_naive(B, A), expected_BA);
  EXPECT_NE(multiply_naive(A, B), multiply_naive(B, A));
}

// ---- Vector shapes ---------------------------------------------------------------

TEST(MultiplyNaive, RowTimesColumnIsDotProduct) {
  // (1x3) * (3x1) -> 1x1:  1*4 + 2*5 + 3*6 = 32
  Matrix row{{1, 2, 3}};
  Matrix col{{4}, {5}, {6}};

  Matrix C = multiply_naive(row, col);

  EXPECT_EQ(C, (Matrix{{32}}));
}

TEST(MultiplyNaive, ColumnTimesRowIsOuterProduct) {
  // (3x1) * (1x2) -> 3x2:  C(i,j) = col(i) * row(j)
  Matrix col{{1}, {2}, {3}};
  Matrix row{{4, 5}};
  Matrix expected{{4, 5},
                  {8, 10},
                  {12, 15}};

  EXPECT_EQ(multiply_naive(col, row), expected);
}

TEST(MultiplyNaive, OneByOne) {
  EXPECT_EQ(multiply_naive(Matrix{{3}}, Matrix{{-4}}), (Matrix{{-12}}));
}

// ---- Special matrices --------------------------------------------------------------

TEST(MultiplyNaive, IdentityLeavesMatrixUnchanged) {
  // A is 2x3. A * I needs I to be 3x3 (A's cols); I * A needs I to be 2x2 (A's rows).
  Matrix A{{1, -2, 3},
           {4, 5, -6}};

  EXPECT_EQ(multiply_naive(A, Matrix::identity(3)), A);
  EXPECT_EQ(multiply_naive(Matrix::identity(2), A), A);
}

TEST(MultiplyNaive, ZeroMatrixGivesZero) {
  Matrix A = make_int_matrix(3, 4, 1);
  Matrix Z(4, 2);  // all zeros

  EXPECT_EQ(multiply_naive(A, Z), Matrix(3, 2));
}

TEST(MultiplyNaive, EmptyInnerDimensionGivesZeros) {
  // (2x0) * (0x3) -> 2x3. Every dot product has zero terms, so every entry is 0.
  Matrix A(2, 0);
  Matrix B(0, 3);

  Matrix C = multiply_naive(A, B);

  ASSERT_EQ(C.rows(), 2u);
  ASSERT_EQ(C.cols(), 3u);
  EXPECT_EQ(C, Matrix(2, 3));
}

// ---- Algebraic properties on bigger, non-square inputs ------------------------------
// These catch index mix-ups (e.g. using j where p belongs) that tiny square
// examples can miss, because every dimension here is different.

TEST(MultiplyNaive, TransposeOfProduct) {
  // (A * B)^T == B^T * A^T
  Matrix A = make_int_matrix(3, 5, 1);
  Matrix B = make_int_matrix(5, 4, 2);

  EXPECT_EQ(transpose(multiply_naive(A, B)),
            multiply_naive(transpose(B), transpose(A)));
}

TEST(MultiplyNaive, DistributesOverAddition) {
  // A * (B + C) == A*B + A*C
  Matrix A = make_int_matrix(4, 3, 3);
  Matrix B = make_int_matrix(3, 6, 4);
  Matrix C = make_int_matrix(3, 6, 5);

  EXPECT_EQ(multiply_naive(A, B + C), multiply_naive(A, B) + multiply_naive(A, C));
}

TEST(MultiplyNaive, Associative) {
  // (A * B) * C == A * (B * C)   (exact here because everything is a small integer)
  Matrix A = make_int_matrix(2, 3, 6);
  Matrix B = make_int_matrix(3, 4, 7);
  Matrix C = make_int_matrix(4, 5, 8);

  EXPECT_EQ(multiply_naive(multiply_naive(A, B), C),
            multiply_naive(A, multiply_naive(B, C)));
}

// ---- Errors ---------------------------------------------------------------------------

TEST(MultiplyNaive, ShapeMismatchThrows) {
  // (2x3) * (2x3): A has 3 columns but B has 2 rows -> undefined product.
  Matrix A(2, 3);
  Matrix B(2, 3);

  EXPECT_THROW(multiply_naive(A, B), std::invalid_argument);
}

TEST(MultiplyNaive, ShapeMismatchThrowsEvenWhenReverseWorks) {
  // (3x2) * (3x2) is invalid, even though both are the "same shape".
  // (2x3) * (3x2) would be fine; order matters.
  Matrix A(3, 2);
  Matrix B(3, 2);
  EXPECT_THROW(multiply_naive(A, B), std::invalid_argument);
  EXPECT_NO_THROW(multiply_naive(transpose(A), B));
}
