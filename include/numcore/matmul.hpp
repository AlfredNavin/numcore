#pragma once

// ----------------------------------------------------------------------------
// HEADER = the "menu". It only DECLARES what exists: name, inputs, output.
// No function bodies here. The bodies live in src/matmul.cpp.
//
// Anyone who wants to multiply matrices writes
//     #include "numcore/matmul.hpp"
// and can call the functions below without seeing how they work.
// ----------------------------------------------------------------------------

#include "numcore/matrix.hpp"  // we need the Matrix type for the signatures

namespace numcore {

// C = A * B using the textbook i-j-k triple loop.  (P1 version 1)
//
//   A is (n x k), B is (k x m)  ->  C is (n x m)
//   C(i, j) = sum over p of A(i, p) * B(p, j)
//
// Throws std::invalid_argument if A.cols() != B.rows().
//
// Anatomy of a declaration:
//   Matrix            -> return type (what the function gives back)
//   multiply_naive    -> function name
//   const Matrix& A   -> parameter: a read-only reference to the caller's
//                        matrix (no copy is made, and we promise not to change it)
//   ;                 -> ends the declaration (no body here!)
Matrix multiply_naive(const Matrix& A, const Matrix& B);

// Later P1 versions get declared here too, e.g.:
// Matrix multiply_ikj(const Matrix& A, const Matrix& B);      // version 2
// Matrix multiply_tiled(const Matrix& A, const Matrix& B);    // version 3

}  // namespace numcore
