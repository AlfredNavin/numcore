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

// C = A * B with the two inner loops swapped: i-k-j order.  (P1 version 2)
//
// Same result as multiply_naive (each C(i,j) still adds its terms in the
// same p order), but the innermost loop now walks along a ROW of B and a ROW
// of C, which sit next to each other in memory.
//
// Throws std::invalid_argument if A.cols() != B.rows().
Matrix multiply_ikj(const Matrix& A, const Matrix& B);

// C = A * B using cache blocking (tiling).  (P1 version 3)
//
// Splits the work into tile x tile blocks small enough to stay in the fast
// caches, and finishes all the work on one block of C before moving on, so
// each number loaded from memory gets reused many times while it's still
// close to the CPU. Inside a block it uses the i-k-j order from version 2.
//
// `tile` is the block edge length in elements. Three tile x tile blocks of
// doubles need 3 * tile^2 * 8 bytes, so roughly:
//   tile = 32  -> 24 KB  (fits L1, 48 KB on this machine)
//   tile = 128 -> 384 KB (fits L2, 1.25 MB on this machine)
// Default 128: the most consistent in benchmarks on this machine (i7-12700H,
// Release, no -march=native). Re-check after version 4 changes the compiler flags.
// Matrix sizes don't need to be multiples of `tile`: edge blocks are smaller.
//
// Same result as multiply_naive (each C(i,j) still adds its terms in the
// same p order).
//
// Throws std::invalid_argument if A.cols() != B.rows() or tile == 0.
Matrix multiply_tiled(const Matrix& A, const Matrix& B, std::size_t tile = 128);

// C = A * B on several CPU cores at once, using OpenMP.  (P1 version 5)
//
// The same tiled algorithm as version 3, but the tile x tile blocks of C are
// shared out between threads. Each block of C is computed by exactly one
// thread, so no two threads ever write the same number: no locks needed.
//
// `threads` = how many threads to use. 0 (the default) means "let OpenMP
// decide", which is normally one per logical CPU (20 on this machine), or
// whatever the OMP_NUM_THREADS environment variable says.
//
// Each C(i,j) is computed by one thread, adding its terms in the same p order
// as multiply_naive, so the result doesn't depend on the thread count, bit for
// bit. (Versus naive: exact for integers; with -march=native, real inputs can
// differ in the last bit because of FMA, see the tests.)
//
// Throws std::invalid_argument if A.cols() != B.rows(), tile == 0 or
// threads < 0.
Matrix multiply_parallel(const Matrix& A, const Matrix& B, std::size_t tile = 128,
                         int threads = 0);

}  // namespace numcore
