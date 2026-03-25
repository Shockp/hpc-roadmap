#include <iostream>
#include <utility>

#include "matrix.h"

memorymanager::Matrix CreateRandomMatrix(int rows, int cols) {
  memorymanager::Matrix temp(rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      temp(i, j) = (i + 1) * (j + 1);
    }
  }
  return temp;
}

int main() {
  std::cout << "--- Creating matrices ---\n";
  memorymanager::Matrix m1 = CreateRandomMatrix(2, 2);
  memorymanager::Matrix m2 = CreateRandomMatrix(2, 2);

  std::cout << "\n--- Testing Copy Constructor ---\n";
  memorymanager::Matrix m3(m1);

  std::cout << "\n--- Testing Move Constructor ---\n";
  memorymanager::Matrix m4(std::move(m3));

  std::cout << "\n--- Testing Move Assignment ---\n";
  memorymanager::Matrix m5;

  m5 = CreateRandomMatrix(3, 3);

  std::cout << "\n--- Testing Matrix Math ---\n";
  memorymanager::Matrix m6 = m1 + m2;
  memorymanager::Matrix m7 = m1 * m2;

  std::cout << "Addition Result:\n";
  m6.Print();

  std::cout << "Multiplication Result:\n";
  m7.Print();

  return 0;
}