// ----------------------------------------------------------------------------
// SOURCE FILE = the "kitchen". This is where the function bodies live.
//
// Rule: a .cpp file includes its OWN header first. If the header forgot an
// #include it needs, this makes the compiler complain here, right away.
// ----------------------------------------------------------------------------
#include "numcore/matmul.hpp"

#include <cstddef>    // std::size_t
#include <stdexcept>  // std::invalid_argument
#include <string>     // std::string, std::to_string

namespace numcore {

// ---- Shared helpers ----------------------------------------------------------
// An unnamed namespace makes everything inside it PRIVATE to this .cpp file
// ("internal linkage"). Other files can't see or call it, and it doesn't go
// in the header, because it's an implementation detail, not part of the
// library's public menu.
namespace {

// Every multiply version must check the same rule before doing any work:
// A * B only exists when A's column count equals B's row count.
//   (n x k) * (k x m) -> (n x m)
// Throws std::invalid_argument with both shapes in the message, e.g.
//   "multiply: cannot multiply 2x3 by 2x3 (A.cols() must equal B.rows())"
void check_multiply_shapes(const Matrix& A, const Matrix& B) {
  if (A.cols() != B.rows()) {
    throw std::invalid_argument(
        "multiply: cannot multiply " + std::to_string(A.rows()) + "x" +
        std::to_string(A.cols()) + " by " + std::to_string(B.rows()) + "x" +
        std::to_string(B.cols()) + " (A.cols() must equal B.rows())");
  }
}

}  // namespace

// The definition must match the declaration in matmul.hpp exactly:
// same return type, same name, same parameter types.
Matrix multiply_naive(const Matrix& A, const Matrix& B) {
  // ---- 1. Read the shapes into named local variables ------------------------
  // `const` = this variable never changes after this line. Use it whenever
  // you can: it documents intent and the compiler stops you changing it.
  // `std::size_t` = unsigned integer type for sizes (like int, but can't be
  // negative and is big enough for any array size).
  const std::size_t n = A.rows();  // rows of A    = rows of C
  const std::size_t k = A.cols();  // cols of A    = must equal rows of B
  const std::size_t m = B.cols();  // cols of B    = cols of C

  // ---- 2. Validate the input ----------------------------------------------
  check_multiply_shapes(A, B);

  // ---- 3. Create the output -------------------------------------------------
  // Matrix(rows, cols) creates a zero-filled matrix (see matrix.hpp line 38).
  // Not const: we're going to write into it.
  Matrix C(n, m);

  // ---- 4. The triple loop ---------------------------------------------------
  // Loop variables are std::size_t to match n, k, m. (Comparing an int with
  // a size_t triggers a -Wsign-compare warning.)
  //
  // Accessor: the fast m(i, j), not m.at(i, j). The shape check plus the loop
  // limits already guarantee every index is in range, so .at()'s bounds check
  // would just cost time and block SIMD. Every multiply version uses m(i, j),
  // so benchmarks compare the algorithms, not the accessors.
  for (std::size_t i = 0; i < n; ++i) {      // each row of C
    for (std::size_t j = 0; j < m; ++j) {    // each column of C
      for (std::size_t p = 0; p < k; ++p) {  // walk row i of A / column j of B
        C(i, j) += A(i, p) * B(p, j);
      }
    }
  }

  // ---- 5. Return the result -------------------------------------------------
  // Returning a local Matrix by value is cheap: the vector's memory is moved
  // out, not copied.
  return C;
}

// ============================================================================
// Version 2: i-k-j loop order
// ============================================================================
//
// The maths is identical to multiply_naive. Only the ORDER of the loops changes:
//
//   naive:  for i -> for j -> for p :  C(i,j) += A(i,p) * B(p,j)
//   i-k-j:  for i -> for p -> for j :  C(i,j) += A(i,p) * B(p,j)
//
// (The plan calls the middle index k; here it's p, because k is already the
// name of A's column count.)
//
// Why this is faster (row-major storage, 64-byte cache lines = 8 doubles):
//   - naive's innermost loop moves p, so it reads B(0,j), B(1,j), B(2,j)...
//     = DOWN a column of B. Each step jumps a whole row ahead in memory, so
//     every read loads a new cache line and uses only 1 of its 8 doubles.
//   - i-k-j's innermost loop moves j, so it reads B(p,0), B(p,1), B(p,2)...
//     and writes C(i,0), C(i,1), C(i,2)... = ALONG rows. Neighbours in
//     memory, so all 8 doubles of each cache line get used.
//   - A(i,p) doesn't depend on j, so it's the same for the whole inner loop:
//     read it once, before the inner loop starts.
Matrix multiply_ikj(const Matrix& A, const Matrix& B) {
  // ---- 1. Shapes (same as naive) --------------------------------------------
  const std::size_t n = A.rows();
  const std::size_t k = A.cols();
  const std::size_t m = B.cols();

  // ---- 2. Validate the input (same rule as naive) ---------------------------
  check_multiply_shapes(A, B);

  // ---- 3. Create the output (zero-filled) ------------------------------------
  Matrix C(n, m);

  // ---- 4. The triple loop, i-k-j order ---------------------------------------
  // Same fast m(i, j) accessor as multiply_naive, so the benchmark compares
  // loop order only.
  for (std::size_t i = 0; i < n; ++i) {        // each row of C (and of A)
    for (std::size_t p = 0; p < k; ++p) {      // each column of A / row of B
      // A(i, p) is the same for every j: read it once, keep it in a register.
      const double a = A(i, p);
      for (std::size_t j = 0; j < m; ++j) {    // walk ALONG row p of B and row i of C
        C(i, j) += a * B(p, j);
      }
    }
  }

  // ---- 5. Return the result ----------------------------------------------------
  return C;
}

}  // namespace numcore
