// ----------------------------------------------------------------------------
// SOURCE FILE = the "kitchen". This is where the function bodies live.
//
// Rule: a .cpp file includes its OWN header first. If the header forgot an
// #include it needs, this makes the compiler complain here, right away.
// ----------------------------------------------------------------------------
#include "numcore/matmul.hpp"

#include <cstddef>    // std::size_t
#include <stdexcept>  // std::invalid_argument

namespace numcore {

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
  const std::size_t l = B.rows();

  // ---- 2. Validate the input ----------------------------------------------
  if (l != k) {
    throw std::invalid_argument("Matrix shape cannot be multiplied");
  }

  // ---- 3. Create the output -------------------------------------------------
  // Matrix(rows, cols) creates a zero-filled matrix (see matrix.hpp line 38).
  // Not const: we're going to write into it.
  Matrix C(n, m);

  // ---- 4. The triple loop ---------------------------------------------------
  // Loop variables are std::size_t to match n, k, m. (Comparing an int with
  // a size_t triggers a -Wsign-compare warning.)
  for (std::size_t i = 0; i < n; ++i) {      // each row of C
    for (std::size_t j = 0; j < m; ++j) {    // each column of C
      for (std::size_t p = 0; p < k; ++p) {  // walk row i of A / column j of B
          C.at(i,j) += A.at(i,p) * B.at(p,j);
      }
    }
  }

  // ---- 5. Return the result -------------------------------------------------
  // Returning a local Matrix by value is cheap: the vector's memory is moved
  // out, not copied.
  return C;
}

}  // namespace numcore
