#ifndef UNIVERSITY_OPENMP_MODELS_COLUMN_STATS_H_
#define UNIVERSITY_OPENMP_MODELS_COLUMN_STATS_H_

namespace openmp {

/**
 * @brief Represents statistical information for a single column (feature) of
 * the dataset.
 */
struct Column_stats {
  float min = 0.0f;      /**< Minimum value in the column. */
  float max = 0.0f;      /**< Maximum value in the column. */
  float mean = 0.0f;     /**< Average value of the column. */
  float variance = 0.0f; /**< Variance of the column. */
};

} // namespace openmp

#endif // UNIVERSITY_OPENMP_MODELS_COLUMN_STATS_H_
