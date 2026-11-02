#pragma once  // Include this header at most once per .cpp file (replaces #ifndef guards).

#include <algorithm>         // std::max
#include <cmath>             // std::abs
#include <cstddef>           // std::size_t
#include <initializer_list>  // brace-list constructor
#include <stdexcept>         // std::invalid_argument, std::out_of_range
#include <vector>

// Everything in the library lives in the `numcore` namespace, so our names
// (Matrix, transpose, ...) never clash with names from users' code or other
// libraries. In competitive programming you'd `using namespace std;` —
// in a library header you never do that, because it leaks into every file
// that includes the header.
namespace numcore {

// A dense matrix of doubles, stored row-major in one contiguous block.
//
// Row-major means row 0 comes first, then row 1, and so on:
//
//   [ a b c ]
//   [ d e f ]   is stored as   a b c d e f
//
// so element (i, j) lives at index  i * cols + j.
//
// Why one flat vector instead of vector<vector<double>>?
//   - One allocation instead of `rows` separate ones.
//   - Rows sit next to each other in memory, which is what makes the CPU
//     cache work well. That's the whole point of P1.
class Matrix {
 public:
  // ---- Construction --------------------------------------------------------

  // Empty 0x0 matrix. `= default` lets the compiler write it for us.
  Matrix() = default;

  // rows x cols matrix, every element set to `fill` (0 by default).
  Matrix(std::size_t rows, std::size_t cols, double fill = 0.0)
      : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

  // Lets you write matrices directly in code, which makes tests readable:
  //   Matrix m{{1, 2, 3},
  //            {4, 5, 6}};
  // Throws if the rows have different lengths.
  Matrix(std::initializer_list<std::initializer_list<double>> rows)
      : rows_(rows.size()), cols_(rows.size() == 0 ? 0 : rows.begin()->size()) {
    data_.reserve(rows_ * cols_);
    for (const auto& row : rows) {
      if (row.size() != cols_) {
        throw std::invalid_argument("Matrix: all rows must have the same length");
      }
      data_.insert(data_.end(), row.begin(), row.end());
    }
  }

  // n x n identity matrix. `static` = called on the class, not an object:
  //   Matrix I = Matrix::identity(3);
  static Matrix identity(std::size_t n) {
    Matrix m(n, n);
    for (std::size_t i = 0; i < n; ++i) m(i, i) = 1.0;
    return m;
  }

  // ---- Shape ---------------------------------------------------------------
  // `const` after the () means "doesn't modify the matrix", so these can be
  // called on const matrices. `[[nodiscard]]` makes the compiler warn if you
  // call m.rows(); and ignore the result (almost certainly a bug).

  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
  [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }

  // ---- Element access ------------------------------------------------------

  // m(i, j): fast, NO bounds check. This is what hot loops (multiply) use.
  // Two versions: the non-const one returns a reference so you can write
  // m(i, j) = 5; the const one lets you read from a const Matrix.
  double& operator()(std::size_t i, std::size_t j) noexcept {
    return data_[i * cols_ + j];
  }
  const double& operator()(std::size_t i, std::size_t j) const noexcept {
    return data_[i * cols_ + j];
  }

  // m.at(i, j): same, but throws std::out_of_range on a bad index.
  // Use it when correctness matters more than speed (tests, user input).
  double& at(std::size_t i, std::size_t j) {
    check_index(i, j);
    return (*this)(i, j);
  }
  const double& at(std::size_t i, std::size_t j) const {
    check_index(i, j);
    return (*this)(i, j);
  }

  // Raw pointer to the first element. Needed later for passing data to
  // BLAS/Eigen (version 6 of P1) and to the GPU (P9–P10).
  double* data() noexcept { return data_.data(); }
  const double* data() const noexcept { return data_.data(); }

  // ---- Arithmetic ----------------------------------------------------------

  // a += b, element by element. Shapes must match.
  Matrix& operator+=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("Matrix +=: shape mismatch");
    }
    // Same shape => same flat layout, so one simple loop covers everything.
    for (std::size_t k = 0; k < data_.size(); ++k) data_[k] += other.data_[k];
    return *this;  // returning *this allows chaining: (a += b) += c
  }

  // ---- Comparison ----------------------------------------------------------

  // Exact equality: same shape and every element bit-for-bit equal.
  // `= default` compares all members in order (rows_, cols_, data_).
  // For results of floating-point arithmetic, use approx_equal() instead.
  bool operator==(const Matrix&) const = default;

 private:
  void check_index(std::size_t i, std::size_t j) const {
    if (i >= rows_ || j >= cols_) throw std::out_of_range("Matrix: index out of range");
  }

  // Trailing underscore = "this is a member variable" (a common convention).
  // `= 0` gives safe defaults for the default constructor.
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<double> data_;
};

// ---- Free functions ----------------------------------------------------------
// These are outside the class because they only need the public interface.
// `inline` is required for functions defined in a header: without it, two
// .cpp files including this header would both define the function and the
// linker would complain ("multiple definition").

// a + b. Takes `a` by value (a copy), adds b into the copy, returns it.
inline Matrix operator+(Matrix a, const Matrix& b) {
  a += b;
  return a;
}

// Returns the transpose: result(j, i) = m(i, j). A 2x3 becomes a 3x2.
inline Matrix transpose(const Matrix& m) {
  Matrix t(m.cols(), m.rows());
  for (std::size_t i = 0; i < m.rows(); ++i)
    for (std::size_t j = 0; j < m.cols(); ++j) t(j, i) = m(i, j);
  return t;
}

// True if a and b have the same shape and every pair of elements is close:
//
//   |x - y| <= abs_tol + rel_tol * max(|x|, |y|)
//
// Why not just ==? Floating-point addition isn't associative:
// (0.1 + 0.2) + 0.3 != 0.1 + (0.2 + 0.3). Your 6 multiply versions add
// things in different orders, so their results differ in the last few bits.
// rel_tol handles large numbers, abs_tol handles values near zero.
inline bool approx_equal(const Matrix& a, const Matrix& b, double rel_tol = 1e-12,
                         double abs_tol = 1e-12) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
  const double* x = a.data();
  const double* y = b.data();
  for (std::size_t k = 0; k < a.size(); ++k) {
    const double diff = std::abs(x[k] - y[k]);
    const double scale = std::max(std::abs(x[k]), std::abs(y[k]));
    // Written as !(diff <= ...) rather than (diff > ...) so NaN counts as
    // "not equal": every comparison involving NaN is false.
    if (!(diff <= abs_tol + rel_tol * scale)) return false;
  }
  return true;
}

}  // namespace numcore
