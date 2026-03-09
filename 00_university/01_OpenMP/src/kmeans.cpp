#include "../include/kmeans.h"

#include <cstdint>
#include <limits>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <vector>

namespace openmp {

Centroids InitializeCentroids(const Dataset &local_data, uint32_t num_clusters,
                              uint64_t global_offset, uint64_t total_rows,
                              MPI_Comm comm) {
  const uint32_t num_rows = local_data.n_rows;
  const uint32_t num_cols = local_data.n_cols;

  // Prepare the Centroids structure that will hold K centroids, each with
  // `num_cols` dimensions, plus a count of rows assigned to each centroid.
  Centroids centroids;
  centroids.num_clusters = num_clusters;
  centroids.num_cols = num_cols;
  centroids.data.resize(num_clusters * num_cols, 0.0f);
  centroids.counts.resize(num_clusters, 0);

  // Thread-shared accumulators for per-cluster sums and counts.
  // `double` is used for sums to reduce floating-point rounding error when
  // accumulating many `float` values.
  std::vector<double> local_sums(num_clusters * num_cols, 0.0);
  std::vector<uint32_t> local_counts(num_clusters, 0);

  // Each row is assigned to a cluster by dividing the global row space evenly
  // among K clusters (global_index / rows_per_cluster).
  uint64_t rows_per_cluster = total_rows / num_clusters;
  if (rows_per_cluster == 0) {
    rows_per_cluster = 1;
  }

  // --- Phase 1: Local accumulation (OpenMP-parallelised) -------------------
  // Each thread maintains private sum/count buffers to avoid contention.
  // After the loop, private buffers are merged into the shared `local_sums`
  // and `local_counts` via a critical section.
#pragma omp parallel
  {
    std::vector<double> thread_sums(num_clusters * num_cols, 0.0);
    std::vector<uint32_t> thread_counts(num_clusters, 0);

#pragma omp for nowait
    for (uint32_t r = 0; r < num_rows; ++r) {
      // Map the local row to its global index to determine its cluster.
      uint64_t global_index = global_offset + r;

      uint32_t cluster_id =
          std::min(static_cast<uint32_t>(global_index / rows_per_cluster),
                   num_clusters - 1);

      thread_counts[cluster_id]++;

      // Accumulate the row's feature values into the cluster's sum buffer.
      const float *row_ptr = local_data.GetRowPtr(r);
      for (uint32_t c = 0; c < num_cols; ++c) {
        thread_sums[cluster_id * num_cols + c] += row_ptr[c];
      }
    }

    // Merge thread-private results into the shared local accumulators.
#pragma omp critical
    {
      for (uint32_t k = 0; k < num_clusters; ++k) {
        local_counts[k] += thread_counts[k];
        for (uint32_t c = 0; c < num_cols; ++c) {
          local_sums[k * num_cols + c] += thread_sums[k * num_cols + c];
        }
      }
    }
  }

  // --- Phase 2: Global aggregation (MPI_Allreduce) -------------------------
  // Combine partial sums and counts from every MPI rank so all ranks hold
  // the same global totals.
  std::vector<double> global_sums(num_clusters * num_cols, 0.0);

  MPI_Allreduce(local_sums.data(), global_sums.data(), num_clusters * num_cols,
                MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(local_counts.data(), centroids.counts.data(), num_clusters,
                MPI_UINT32_T, MPI_SUM, comm);

  // --- Phase 3: Compute centroid positions ---------------------------------
  // Each centroid = global_sum / global_count for its cluster.
  for (uint32_t k = 0; k < num_clusters; ++k) {
    uint32_t count = centroids.counts[k];
    if (count > 0) {
      for (uint32_t c = 0; c < num_cols; ++c) {
        centroids.data[k * num_cols + c] = static_cast<float>(
            global_sums[k * num_cols + c] / static_cast<double>(count));
      }
    }
  }

  return centroids;
}

Centroids RunKMeans(Dataset &local_data, uint32_t num_clusters,
                    uint32_t max_iterations, uint64_t global_offset,
                    uint64_t total_rows, MPI_Comm comm) {
  int rank = 0;
  int num_procs = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &num_procs);

  const uint32_t num_cols = local_data.n_cols;

  // --- Step 0: Boot-strap centroids from an equal-partition average ---------
  Centroids centroids = InitializeCentroids(local_data, num_clusters,
                                            global_offset, total_rows, comm);

  // Initialise assignments using the same partition scheme as
  // InitializeCentroids so the first iteration starts from a consistent state.
  std::vector<uint32_t> local_assignments(local_data.n_rows, 0);
  uint64_t rows_per_cluster = total_rows / num_clusters;
  if (rows_per_cluster == 0)
    rows_per_cluster = 1;

  for (uint32_t r = 0; r < local_data.n_rows; ++r) {
    uint64_t global_index = global_offset + r;
    local_assignments[r] =
        std::min(static_cast<uint32_t>(global_index / rows_per_cluster),
                 num_clusters - 1);
  }

  uint32_t iteration = 0;
  bool converged = false;

  // ========================= Main K-Means loop =============================
  while (iteration < max_iterations && !converged) {
    uint64_t local_displacements = 0;
    uint32_t current_num_rows = local_data.n_rows;

    // --- Step 1: Assignment (OpenMP-parallelised) --------------------------
    // Each row is reassigned to the nearest centroid using squared Euclidean
    // distance. Thread-private displacement counters are merged atomically.
#pragma omp parallel
    {
      uint64_t thread_displacement = 0;

#pragma omp for nowait
      for (uint32_t r = 0; r < current_num_rows; ++r) {
        const float *row_ptr = local_data.GetRowPtr(r);

        float min_dist_sq = std::numeric_limits<float>::max();
        uint32_t best_cluster = local_assignments[r];

        // Find the closest centroid to this row.
        for (uint32_t k = 0; k < num_clusters; ++k) {
          const float *centroid_ptr = centroids.GetClusterPtr(k);
          float dist_sq = 0.0f;

          for (uint32_t c = 0; c < num_cols; ++c) {
            float diff = row_ptr[c] - centroid_ptr[c];
            dist_sq += diff * diff;
          }

          if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            best_cluster = k;
          }
        }

        // Track changes for the convergence check.
        if (best_cluster != local_assignments[r]) {
          local_assignments[r] = best_cluster;
          thread_displacement++;
        }
      }

#pragma omp atomic
      local_displacements += thread_displacement;
    }

    // --- Step 2: Convergence check (MPI_Allreduce) -------------------------
    // If fewer than 5 % of all rows changed cluster, declare convergence.
    uint64_t global_displacement = 0;
    MPI_Allreduce(&local_displacements, &global_displacement, 1, MPI_UINT64_T,
                  MPI_SUM, comm);

    double displacement_ratio = static_cast<double>(global_displacement) /
                                static_cast<double>(total_rows);

    if (displacement_ratio < 0.05) {
      converged = true;
      break;
    }

    // --- Step 3: Data redistribution (MPI_Alltoallv) -----------------------
    // Each cluster is "owned" by exactly one MPI rank. Rows that now belong
    // to a cluster on a different rank are shipped there. This keeps the data
    // locality aligned with the cluster ownership.

    // Lambda: determines which MPI rank owns a given cluster_id by dividing
    // the clusters as evenly as possible across ranks.
    auto get_owner_rank = [&](uint32_t cluster_id) -> int {
      uint32_t base_clusters = num_clusters / static_cast<uint32_t>(num_procs);
      uint32_t remainder = num_clusters % static_cast<uint32_t>(num_procs);

      if (base_clusters == 0) {
        return static_cast<int>(cluster_id % static_cast<uint32_t>(num_procs));
      }

      int rank_owner = 0;
      uint32_t current_cluster_limit = 0;
      for (int i = 0; i < num_procs; ++i) {
        current_cluster_limit +=
            base_clusters + (static_cast<uint32_t>(i) < remainder ? 1 : 0);
        if (cluster_id < current_cluster_limit) {
          rank_owner = i;
          break;
        }
      }
      return rank_owner;
    };

    // Partition rows: keep rows owned by this rank, buffer the rest for
    // their destination rank.
    std::vector<int> send_point_counts(num_procs, 0);
    std::vector<std::vector<float>> send_data_buffers(num_procs);
    std::vector<std::vector<uint32_t>> send_assignment_buffers(num_procs);

    std::vector<float> next_local_data;
    std::vector<uint32_t> next_local_assignments;
    next_local_data.reserve(local_data.data.size());
    next_local_assignments.reserve(local_assignments.size());

    for (uint32_t r = 0; r < current_num_rows; ++r) {
      uint32_t cluster_id = local_assignments[r];
      int owner_rank = get_owner_rank(cluster_id);

      const float *row_ptr = local_data.GetRowPtr(r);

      if (owner_rank == rank) {
        // Row stays on this rank.
        next_local_data.insert(next_local_data.end(), row_ptr,
                               row_ptr + num_cols);
        next_local_assignments.push_back(cluster_id);
      } else {
        // Row needs to be sent to `owner_rank`.
        send_data_buffers[owner_rank].insert(
            send_data_buffers[owner_rank].end(), row_ptr, row_ptr + num_cols);
        send_assignment_buffers[owner_rank].push_back(cluster_id);
        send_point_counts[owner_rank]++;
      }
    }

    // Flatten per-rank send buffers into contiguous arrays and compute
    // displacement/count arrays required by MPI_Alltoallv.
    std::vector<float> flat_send_data;
    std::vector<uint32_t> flat_send_assignments;
    std::vector<int> send_data_displacements(num_procs, 0);
    std::vector<int> send_data_counts(num_procs, 0);
    std::vector<int> send_assign_displacements(num_procs, 0);

    int current_data_disp = 0;
    int current_assign_disp = 0;

    for (int i = 0; i < num_procs; ++i) {
      send_data_displacements[i] = current_data_disp;
      send_data_counts[i] = send_point_counts[i] * num_cols;
      current_data_disp += send_data_counts[i];
      flat_send_data.insert(flat_send_data.end(), send_data_buffers[i].begin(),
                            send_data_buffers[i].end());

      send_assign_displacements[i] = current_assign_disp;
      current_assign_disp += send_point_counts[i];
      flat_send_assignments.insert(flat_send_assignments.end(),
                                   send_assignment_buffers[i].begin(),
                                   send_assignment_buffers[i].end());
    }

    // Exchange point counts so each rank knows how much data it will receive.
    std::vector<int> recv_point_counts(num_procs, 0);
    MPI_Alltoall(send_point_counts.data(), 1, MPI_INT, recv_point_counts.data(),
                 1, MPI_INT, comm);

    // Build receive displacement/count arrays.
    std::vector<int> recv_data_counts(num_procs, 0);
    std::vector<int> recv_data_displacements(num_procs, 0);
    std::vector<int> recv_assign_displacements(num_procs, 0);

    int total_recv_points = 0;
    int total_recv_floats = 0;

    for (int i = 0; i < num_procs; ++i) {
      recv_assign_displacements[i] = total_recv_points;
      total_recv_points += recv_point_counts[i];

      recv_data_counts[i] = recv_point_counts[i] * num_cols;
      recv_data_displacements[i] = total_recv_floats;
      total_recv_floats += recv_data_counts[i];
    }

    std::vector<float> flat_recv_data(total_recv_floats);
    std::vector<uint32_t> flat_recv_assignments(total_recv_points);

    // Exchange feature data (floats) between all ranks.
    MPI_Alltoallv(flat_send_data.data(), send_data_counts.data(),
                  send_data_displacements.data(), MPI_FLOAT,
                  flat_recv_data.data(), recv_data_counts.data(),
                  recv_data_displacements.data(), MPI_FLOAT, comm);

    // Exchange cluster assignments (uint32_t) between all ranks.
    MPI_Alltoallv(flat_send_assignments.data(), send_point_counts.data(),
                  send_assign_displacements.data(), MPI_UINT32_T,
                  flat_recv_assignments.data(), recv_point_counts.data(),
                  recv_assign_displacements.data(), MPI_UINT32_T, comm);

    // Merge received data with the rows that already stayed on this rank.
    next_local_data.insert(next_local_data.end(), flat_recv_data.begin(),
                           flat_recv_data.end());
    next_local_assignments.insert(next_local_assignments.end(),
                                  flat_recv_assignments.begin(),
                                  flat_recv_assignments.end());

    // Replace the local dataset with the newly redistributed data.
    local_data.data = std::move(next_local_data);
    local_data.n_rows = local_data.data.size() / num_cols;
    local_assignments = std::move(next_local_assignments);

    // --- Step 4: Centroid update (OpenMP + MPI_Allreduce) -------------------
    // Recompute centroids as the mean of all rows assigned to each cluster,
    // using the same thread-private-buffer pattern as InitializeCentroids.
    std::vector<double> local_sums(num_clusters * num_cols, 0.0);
    std::vector<uint32_t> local_counts(num_clusters, 0);

#pragma omp parallel
    {
      std::vector<double> thread_sums(num_clusters * num_cols, 0.0);
      std::vector<uint32_t> thread_counts(num_clusters, 0);

#pragma omp for nowait
      for (uint32_t r = 0; r < local_data.n_rows; ++r) {
        uint32_t cluster_id = local_assignments[r];
        thread_counts[cluster_id]++;
        const float *row_ptr = local_data.GetRowPtr(r);
        for (uint32_t c = 0; c < num_cols; ++c) {
          thread_sums[cluster_id * num_cols + c] += row_ptr[c];
        }
      }

      // Merge thread-private sums/counts into the shared local accumulators.
#pragma omp critical
      {
        for (uint32_t k = 0; k < num_clusters; ++k) {
          local_counts[k] += thread_counts[k];
          for (uint32_t c = 0; c < num_cols; ++c) {
            local_sums[k * num_cols + c] += thread_sums[k * num_cols + c];
          }
        }
      }
    }

    // Combine local sums and counts from all MPI ranks.
    std::vector<double> global_sums(num_clusters * num_cols, 0.0);
    MPI_Allreduce(local_sums.data(), global_sums.data(),
                  num_clusters * num_cols, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(local_counts.data(), centroids.counts.data(), num_clusters,
                  MPI_UINT32_T, MPI_SUM, comm);

    // Derive final centroid positions from the global sums and counts.
    for (uint32_t k = 0; k < num_clusters; ++k) {
      uint32_t count = centroids.counts[k];
      if (count > 0) {
        for (uint32_t c = 0; c < num_cols; ++c) {
          centroids.data[k * num_cols + c] = static_cast<float>(
              global_sums[k * num_cols + c] / static_cast<double>(count));
        }
      }
    }

    iteration++;
  }

  return centroids;
}

} // namespace openmp