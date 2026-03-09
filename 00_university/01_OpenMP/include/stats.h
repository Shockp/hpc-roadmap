#ifndef UNIVERSITY_OPENMP_STATS_H_
#define UNIVERSITY_OPENMP_STATS_H_

#include <vector>

#include <mpi.h>

#include "models/column_stats.h"
#include "models/dataset.h"

namespace openmp {

/**
 * @brief Computes per-column statistics across all MPI ranks.
 *
 * For each column in the dataset this function computes the minimum, maximum,
 * mean, and variance. Row-level accumulation within the local partition is
 * parallelised with OpenMP (using array reductions on sum, sum-of-squares, min,
 * and max). The partial results are then combined across all MPI ranks with
 * MPI_Allreduce so that every rank receives the same global statistics.
 *
 * Variance is calculated with the "mean of squares minus square of the mean"
 * identity and is clamped to zero to guard against negative values that can
 * arise from floating-point rounding.
 *
 * @param local_data The dataset partition owned by the calling MPI rank.
 * @param comm       The MPI communicator to use (defaults to MPI_COMM_WORLD).
 * @return std::vector<openmp::Column_stats> A vector of length
 *         `local_data.n_cols` containing the global statistics for each column.
 *         Returns default-initialised statistics when the global row count is
 *         zero.
 */
std::vector<openmp::Column_stats>
ComputeLocalStats(const openmp::Dataset &local_data, MPI_Comm = MPI_COMM_WORLD);

} // namespace openmp

#endif // UNIVERSITY_OPENMP_STATS_H_