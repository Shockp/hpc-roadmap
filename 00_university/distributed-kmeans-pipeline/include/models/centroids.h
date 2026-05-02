#ifndef UNIVERSITY_OPENMP_MODELS_CENTROIDS_H_
#define UNIVERSITY_OPENMP_MODELS_CENTROIDS_H_

#include <cstdint>
#include <vector>

namespace openmp {

/**
 * @brief Represents the cluster centroids in the K-Means algorithm.
 *
 * Similar to Dataset, this structure uses a flattened 1D vector to store
 * the 2D matrix of centroids for optimal cache performance. It also keeps
 * track of the number of points assigned to each cluster.
 */
struct Centroids {
  uint32_t num_clusters = 0; /**< Total number of clusters (K). */
  uint32_t num_cols = 0;     /**< Number of dimensions/features per centroid. */
  std::vector<float> data; /**< Flattened 1D array storing centroid coordinates
                              in row-major order. */
  std::vector<uint32_t>
      counts; /**< Number of dataset points assigned to each cluster. */

  /**
   * @brief Retrieves a specific coordinate value for a given cluster centroid.
   *
   * @param cluster_idx The 0-based index of the cluster.
   * @param col The 0-based index of the dimension/column.
   * @return The coordinate value.
   */
  inline float GetValue(uint32_t cluster_idx, uint32_t col) const {
    return data[cluster_idx * num_cols + col];
  }

  /**
   * @brief Obtains a read-only pointer to the beginning of a specific
   * centroid's coordinates.
   *
   * @param cluster_idx The 0-based index of the cluster.
   * @return A constant pointer to the first coordinate of the centroid.
   */
  inline const float *GetClusterPtr(uint32_t cluster_idx) const {
    return &data[cluster_idx * num_cols];
  }

  /**
   * @brief Obtains a mutable pointer to the beginning of a specific centroid's
   * coordinates.
   *
   * @param cluster_idx The 0-based index of the cluster.
   * @return A pointer to the first coordinate of the centroid.
   */
  inline float *GetClusterPtr(uint32_t cluster_idx) {
    return &data[cluster_idx * num_cols];
  }
};

} // namespace openmp

#endif // UNIVERSITY_OPENMP_MODELS_CENTROIDS_H_
