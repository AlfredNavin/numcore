#pragma once

// Teaches GoogleTest how to print a Matrix in failure messages, e.g.
//   2x2 [[19, 22], [43, 50]]
// instead of a dump of raw bytes.
//
// GoogleTest finds PrintTo by argument-dependent lookup, so it must live in
// the same namespace as Matrix. Every test file that compares Matrix values
// must include this header: if one file sees PrintTo and another doesn't,
// they compile different versions of the same GoogleTest template and the
// linker silently keeps just one of them (an ODR violation).

#include "numcore/matrix.hpp"

#include <cstddef>
#include <ostream>

namespace numcore {

inline void PrintTo(const Matrix& m, std::ostream* os) {
  *os << m.rows() << "x" << m.cols() << " [";
  for (std::size_t i = 0; i < m.rows(); ++i) {
    *os << (i ? ", [" : "[");
    for (std::size_t j = 0; j < m.cols(); ++j) *os << (j ? ", " : "") << m(i, j);
    *os << "]";
  }
  *os << "]";
}

}  // namespace numcore
