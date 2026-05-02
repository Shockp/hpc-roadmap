#include "../include/stats.h"

#include <mpi.h>
#include <omp.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace openmp {

std::vector<openmp::Column_stats>
ComputeLocalStats(const openmp::Dataset &local_data, MPI_Comm comm) {
  const uint32_t num_rows = local_data.n_rows;
  const uint32_t num_cols = local_data.n_cols;

  // --- Phase 1: Local accumulation (OpenMP-parallelised) -------------------
  // Each thread gets its own copy of these arrays via OpenMP array reductions.
  // `double` is used for sums to minimise floating-point error when
  // accumulating many `float` values.
  std::vector<double> local_sum(num_cols, 0.0);
  std::vector<double> local_sum_sq(num_cols, 0.0);
  std::vector<float> local_min(num_cols, std::numeric_limits<float>::max());
  std::vector<float> local_max(num_cols, std::numeric_limits<float>::lowest());

  // Raw pointers are required for OpenMP user-defined array reductions.
  double *sum_ptr = local_sum.data();
  double *sum_sq_ptr = local_sum_sq.data();
  float *min_ptr = local_min.data();
  float *max_ptr = local_max.data();

  // Rows are distributed across threads; each thread reduces into its private
  // copy of the column-length arrays, which OpenMP merges after the loop.
#pragma omp parallel for reduction(+ : sum_ptr[ : num_cols],                   \
                                       sum_sq_ptr[ : num_cols])                \
    reduction(min : min_ptr[ : num_cols])                                      \
    reduction(max : max_ptr[ : num_cols])
  for (uint32_t r = 0; r < num_rows; ++r) {
    const float *row_ptr = local_data.GetRowPtr(r);

    for (uint32_t c = 0; c < num_cols; ++c) {
      const float val = row_ptr[c];

      sum_ptr[c] += val;
      sum_sq_ptr[c] += static_cast<double>(val) * val;

      min_ptr[c] = std::min(min_ptr[c], val);
      max_ptr[c] = std::max(max_ptr[c], val);
    }
  }

  // --- Phase 2: Global aggregation (MPI_Allreduce) -------------------------
  // Combine partial results from every MPI rank so all ranks hold the same
  // global sums, min, max, and total row count.
  std::vector<double> global_sum(num_cols, 0.0);
  std::vector<double> global_sum_sq(num_cols, 0.0);
  std::vector<float> global_min(num_cols, std::numeric_limits<float>::max());
  std::vector<float> global_max(num_cols, std::numeric_limits<float>::lowest());

  MPI_Allreduce(local_sum.data(), global_sum.data(), num_cols, MPI_DOUBLE,
                MPI_SUM, comm);
  MPI_Allreduce(local_sum_sq.data(), global_sum_sq.data(), num_cols, MPI_DOUBLE,
                MPI_SUM, comm);
  MPI_Allreduce(local_min.data(), global_min.data(), num_cols, MPI_FLOAT,
                MPI_MIN, comm);
  MPI_Allreduce(local_max.data(), global_max.data(), num_cols, MPI_FLOAT,
                MPI_MAX, comm);

  // Total number of rows across all ranks (used to compute the mean).
  uint64_t local_rows_64 = num_rows;
  uint64_t global_rows = 0;
  MPI_Allreduce(&local_rows_64, &global_rows, 1, MPI_UINT64_T, MPI_SUM, comm);

  // --- Phase 3: Derive final statistics -----------------------------------
  std::vector<openmp::Column_stats> results(num_cols);
  if (global_rows == 0) {
    return results;
  }

  for (uint32_t c = 0; c < num_cols; ++c) {
    results[c].min = global_min[c];
    results[c].max = global_max[c];

    // Mean = global_sum / N
    double mean = global_sum[c] / static_cast<double>(global_rows);
    results[c].mean = static_cast<float>(mean);

    // Variance via the computational formula:  Var = E[X²] - (E[X])²
    // Clamped to zero to handle negative values from floating-point rounding.
    double mean_of_squares =
        global_sum_sq[c] / static_cast<double>(global_rows);
    double variance = mean_of_squares - (mean * mean);

    if (variance < 0.0) {
      variance = 0.0;
    }

    results[c].variance = static_cast<float>(variance);
  }

  return results;
}

} // namespace openmp