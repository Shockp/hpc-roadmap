#ifndef UNIVERSITY_OPENMP_MODELS_DATASET_H_
#define UNIVERSITY_OPENMP_MODELS_DATASET_H_

#include <cstdint>
#include <vector>

namespace openmp {

/**
 * @brief Represents a 2D dataset utilizing a flattened 1D vector for optimal
 * memory layout.
 *
 * This structure is intended for high-performance numerical applications,
 * ensuring contiguous memory allocation. This provides better cache locality
 * and enables vectorization or simple bulk memory transfers for MPI/OpenMP.
 */
struct Dataset {
  uint32_t n_rows; /**< Number of rows (samples/data points) in the dataset. */
  uint32_t n_cols; /**< Number of columns (features/dimensions) per sample. */
  std::vector<float>
      data; /**< Flattened 1D array storing the data in row-major order. */

  /**
   * @brief Retrieves a specific value from the dataset via its row and column
   * index.
   *
   * @param row The 0-based index of the row.
   * @param col The 0-based index of the column.
   * @return The float value located at the specified row and column.
   */
  inline float GetValue(uint32_t row, uint32_t col) const {
    return data[row * n_cols + col];
  }

  /**
   * @brief Obtains a read-only pointer to the beginning of a specific row.
   *
   * @param row The 0-based index of the row.
   * @return A constant pointer to the first element of the specified row.
   */
  inline const float *GetRowPtr(uint32_t row) const {
    return &data[row * n_cols];
  }

  /**
   * @brief Obtains a mutable pointer to the beginning of a specific row.
   *
   * @param row The 0-based index of the row.
   * @return A pointer to the first element of the specified row.
   */
  inline float *GetRowPtr(uint32_t row) { return &data[row * n_cols]; }
};

} // namespace openmp

#endif // UNIVERSITY_OPENMP_MODELS_DATASET_H_
