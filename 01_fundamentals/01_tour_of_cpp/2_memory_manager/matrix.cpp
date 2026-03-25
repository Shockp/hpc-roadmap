#include "matrix.h"

#include <algorithm>
#include <cassert>
#include <iostream>

namespace memorymanager {

Matrix::Matrix() : rows_(0), cols_(0), data_(nullptr) {}

Matrix::Matrix(int rows, int cols) : rows_(rows), cols_(cols) {
  data_ = new double[rows_ * cols]{0.0};
}

Matrix::~Matrix() { delete[] data_; }

Matrix::Matrix(const Matrix &other) : rows_(other.rows_), cols_(other.cols_) {
  data_ = new double[rows_ * cols_];
  std::copy(other.data_, other.data_ + (rows_ * cols_), data_);
}

Matrix::Matrix(Matrix &&other) noexcept
    : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {
  other.rows_ = 0;
  other.cols_ = 0;
  other.data_ = nullptr;
}

Matrix &Matrix::operator=(const Matrix &other) {
  if (this == &other) return *this;

  double *new_data = new double[other.rows_ * other.cols_];
  std::copy(other.data_, other.data_ + (other.rows_ + other.cols_), new_data);

  delete[] data_;

  rows_ = other.rows_;
  cols_ = other.cols_;
  data_ = new_data;

  return *this;
}

Matrix &Matrix::operator=(Matrix &&other) noexcept {
  if (this == &other) return *this;

  delete[] data_;

  rows_ = other.rows_;
  cols_ = other.cols_;
  data_ = other.data_;

  other.rows_ = 0;
  other.cols_ = 0;
  other.data_ = nullptr;

  return *this;
}

Matrix Matrix::operator+(const Matrix &other) const {
  assert(rows_ == other.rows_ && cols_ == other.cols_);

  Matrix result(rows_, cols_);
  for (int i = 0; i < rows_ * cols_; ++i) {
    result.data_[i] = data_[i] + other.data_[i];
  }

  return result;
}

Matrix Matrix::operator*(const Matrix &other) const {
  assert(cols_ == other.rows_);

  Matrix result(rows_, other.cols_);

  for (int i = 0; i < rows_; ++i) {
    for (int k = 0; k < cols_; ++k) {
      double a_ik = (*this)(i, k);
      for (int j = 0; j < other.cols_; ++j) {
        result(i, j) += a_ik * other(k, j);
      }
    }
  }
  return result;
}

double &Matrix::operator()(int row, int col) {
  assert(row <= rows_ && col <= cols_);

  return data_[row * cols_ + col];
}

const double &Matrix::operator()(int row, int col) const {
  assert(row <= rows_ && col <= cols_);

  return data_[row * cols_ + col];
}

void Matrix::Print() const {
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      std::cout << (*this)(i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}

}  // namespace memorymanager