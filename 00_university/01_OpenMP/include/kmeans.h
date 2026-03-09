#ifndef UNIVERSITY_OPENMP_KMEANS_H_
#define UNIVERSITY_OPENMP_KMEANS_H_

#include "models/centroids.h"
#include "models/dataset.h"

#include <mpi.h>

#include <cstdint>

namespace openmp {

/**
 * @brief Initialises centroids by partitioning the global dataset into
 *        equal-sized slices and averaging each slice.
 *
 * Every local row is assigned to a cluster based on its global index
 * (global_index / rows_per_cluster). Per-cluster sums are accumulated in
 * parallel with OpenMP (thread-private buffers merged via a critical section)
 * and then combined across all MPI ranks with MPI_Allreduce. Each centroid
 * is set to the mean of its assigned rows.
 *
 * @param local_data     The dataset partition owned by the calling MPI rank.
 * @param num_clusters   The desired number of clusters (K).
 * @param global_offset  The global row index of the first row in local_data.
 * @param total_rows     The total number of rows across all MPI ranks.
 * @param comm           The MPI communicator to use.
 * @return Centroids     A Centroids object with `num_clusters` centroids
 *                       computed as the mean of each partition slice.
 */
Centroids InitializeCentroids(const Dataset &local_data, uint32_t num_clusters,
                              uint64_t global_offset, uint64_t total_rows,
                              MPI_Comm comm);

/**
 * @brief Runs the K-Means clustering algorithm with data redistribution.
 *
 * The algorithm proceeds iteratively:
 *   1. **Assignment** – Each local row is assigned to its nearest centroid
 *      (squared Euclidean distance) in an OpenMP-parallelised loop.
 *   2. **Convergence check** – The global displacement ratio (fraction of
 *      rows that changed cluster) is computed via MPI_Allreduce; the loop
 *      terminates when this ratio drops below 5 % or max_iterations is
 *      reached.
 *   3. **Data redistribution** – Rows are migrated between MPI ranks so that
 *      each rank owns the rows belonging to its assigned cluster range
 *      (MPI_Alltoallv for both feature data and assignment vectors).
 *   4. **Centroid update** – New centroids are computed as the mean of the
 *      assigned rows (same OpenMP + MPI_Allreduce pattern used in
 *      InitializeCentroids).
 *
 * @param local_data      The dataset partition owned by the calling MPI rank.
 *                         Modified in place: after return it contains only the
 *                         rows that belong to the clusters assigned to this
 *                         rank.
 * @param num_clusters     The desired number of clusters (K).
 * @param max_iterations   Maximum number of iterations before stopping.
 * @param global_offset    The global row index of the first row in local_data.
 * @param total_rows       The total number of rows across all MPI ranks.
 * @param comm             The MPI communicator to use.
 * @return Centroids       The final centroids after convergence or after
 *                         max_iterations.
 */
Centroids RunKMeans(Dataset &local_data, uint32_t num_clusters,
                    uint32_t max_iterations, uint64_t global_offset,
                    uint64_t total_rows, MPI_Comm comm);

} // namespace openmp

#endif // UNIVERSITY_OPENMP_KMEANS_H_