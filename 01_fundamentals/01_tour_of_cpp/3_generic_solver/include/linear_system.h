#ifndef GENERIC_SOLVER_LINEAR_SYSTEM_H_
#define GENERIC_SOLVER_LINEAR_SYSTEM_H_

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace genericsolver {

template <typename T>
class LinearSystem {
  static_assert(std::is_floating_point_v<T>,
                "LinearSystem strictly requires a floating-point type.");

 public:
  LinearSystem(int n_variables, std::vector<T> augmented_matrix)
      : n_variables_(n_variables), data_(std::move(augmented_matrix)) {
    if (data_.size() !=
        static_cast<size_t>(n_variables_ * (n_variables_ + 1))) {
      throw std::invalid_argument("Matrix size does not match n * (n+1).");
    }
  }

  // Gaussian Elimination with partial pivoting
  std::vector<T> Solve() {
    for (int k = 0; k < n_variables_; ++k) {
      // Find pivot for numerical stability
      int max_row = k;
      T max_val = std::abs(at(k, k));
      for (int i = k + 1; i < n_variables_; ++i) {
        if (std::abs(at(i, k)) > max_val) {
          max_val = std::abs(at(i, k));
          max_row = i;
        }
      }

      // Swap rows if a vetter pivot was found
      if (max_row != k) {
        for (int j = k; j <= n_variables_; ++j) {
          std::swap(at(k, j), at(max_row, j));
        }
      }

      // Eliminate
      T pivot = at(k, k);
      if (std::abs(pivot) < 1e-9) {
        throw std::runtime_error("System is signular or nearly singular.");
      }

      // Division is slow. Calculate reciprocal once.
      T inv_pivot = static_cast<T>(1.0) / pivot;

      for (int i = k + 1; i < n_variables_; ++i) {
        T factor = at(i, k) * inv_pivot;
        for (int j = k; j <= n_variables_; ++j) {
          at(i, j) -= factor * at(k, j);
        }
      }
    }

    // Back Substitution
    std::vector<T> solution(n_variables_, static_cast<T>(0.0));
    for (int i = n_variables_ - 1; i >= 0; --i) {
      T sum = at(i, n_variables_);
      for (int j = i + 1; j < n_variables_; ++j) {
        sum -= at(i, j) * solution[j];
      }
      solution[i] = sum / at(i, i);
    }

    return solution;
  }

  void Print() const {
    for (int i = 0; i < n_variables_; ++i) {
      for (int j = 0; j <= n_variables_; ++j) {
        std::cout << at(i, j) << "\t";
      }
      std::cout << "\n";
    }
  }

 private:
  int n_variables_;
  std::vector<T> data_;

  T &at(int row, int col) {
    assert(row < n_variables_ && col <= n_variables_);
    return data_[row * (n_variables_ + 1) + col];
  }

  const T &at(int row, int col) const {
    assert(row < n_variables_ && col <= n_variables_);
    return data_[row * (n_variables_ + 1) + col];
  }
};

}  // namespace genericsolver

#endif