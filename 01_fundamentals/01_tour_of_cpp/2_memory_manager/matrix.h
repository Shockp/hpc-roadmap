#ifndef MEMORY_MANAGER_MATRIX_H_
#define MEMORY_MANAGER_MATRIX_H_

namespace memorymanager {

class Matrix {
 public:
  Matrix();
  Matrix(int rows, int cols);

  ~Matrix();

  Matrix(const Matrix &other);
  Matrix(Matrix &&other) noexcept;

  Matrix &operator=(const Matrix &other);
  Matrix &operator=(Matrix &&other) noexcept;

  Matrix operator+(const Matrix &other) const;
  Matrix operator*(const Matrix &other) const;

  double &operator()(int row, int col);
  const double &operator()(int row, int col) const;

  void Print() const;
  int rows() const { return rows_; }
  int cols() const { return cols_; }

 private:
  int rows_;
  int cols_;
  double *data_;
};

}  // namespace memorymanager

#endif