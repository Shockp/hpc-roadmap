#include <mpi.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../include/io_utils.h"
#include "../include/kmeans.h"
#include "../include/models/dataset.h"
#include "../include/stats.h"

/**
 * @brief Main entry point for the MPI + OpenMP HPC pipeline.
 *
 * The program executes three phases on a binary dataset distributed across
 * MPI ranks:
 *   1. **I/O & Scatter** – Rank 0 reads the dataset from disk and distributes
 *      rows to all ranks via MPI_Scatterv (load-balanced for uneven splits).
 *   2. **Column Statistics** – Computes per-column min, max, mean and variance
 *      across all ranks (OpenMP + MPI_Allreduce).
 *   3. **K-Means Clustering** – Runs K-Means with data redistribution until
 *      convergence or `max_iterations` (OpenMP + MPI_Alltoallv).
 *
 * Rank 0 prints a summary of the statistics, final centroids, and wall-clock
 * timings for every phase.
 *
 * @param argc Number of command-line arguments.
 * @param argv Command-line arguments.
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error.
 */
int main(int argc, char **argv) {
  // --- MPI initialisation --------------------------------------------------
  MPI_Init(&argc, &argv);

  int rank = 0;
  int num_procs = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  openmp::Dataset global_dataset;
  uint32_t dimensions[2] = {0, 0};

  // Start wall-clock timers for the total runtime and I/O phase.
  double total_start_time = MPI_Wtime();
  double io_start_time = MPI_Wtime();

  // --- Phase 1: I/O & Scatter ----------------------------------------------
  // Rank 0 reads the binary dataset and broadcasts row/column counts.
  if (rank == 0) {
    try {
      global_dataset = openmp::ReadBinaryFile("../data/dataset.bin").value();
      dimensions[0] = global_dataset.n_rows;
      dimensions[1] = global_dataset.n_cols;
      std::cout << "\n[Rank 0] Successfully loaded " << dimensions[0]
                << " rows and " << dimensions[1] << " columns.\n";
    } catch (const std::exception &e) {
      std::cerr << "[Rank 0] Error: " << e.what() << "\n";
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  // Broadcast dataset dimensions so every rank knows the shape.
  MPI_Bcast(dimensions, 2, MPI_UINT32_T, 0, MPI_COMM_WORLD);

  uint32_t total_rows = dimensions[0];
  uint32_t num_cols = dimensions[1];

  // Distribute rows as evenly as possible; the first `remainder` ranks each
  // receive one extra row.
  int local_rows = total_rows / num_procs;
  int remainder = total_rows % num_procs;
  if (rank < remainder) {
    local_rows++;
  }

  // Compute the global row offset for this rank (needed by K-Means for
  // initial cluster assignment).
  uint64_t global_offset = 0;
  for (int i = 0; i < rank; ++i) {
    global_offset += (total_rows / num_procs) + (i < remainder ? 1 : 0);
  }

  // Build the sendcounts and displacement arrays required by MPI_Scatterv.
  std::vector<int> sendcounts(num_procs, 0);
  std::vector<int> displacements(num_procs, 0);

  if (rank == 0) {
    int current_displacement = 0;
    for (int i = 0; i < num_procs; ++i) {
      int rows_for_proc = total_rows / num_procs + (i < remainder ? 1 : 0);
      sendcounts[i] = rows_for_proc * num_cols;
      displacements[i] = current_displacement;
      current_displacement += sendcounts[i];
    }
  }

  // Allocate storage for the local partition and scatter the data.
  openmp::Dataset local_dataset;
  local_dataset.n_rows = local_rows;
  local_dataset.n_cols = num_cols;
  local_dataset.data.resize(local_rows * num_cols);

  MPI_Scatterv(global_dataset.data.data(), sendcounts.data(),
               displacements.data(), MPI_FLOAT, local_dataset.data.data(),
               local_rows * num_cols, MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  double io_end_time = MPI_Wtime();

  // --- Phase 2: Column Statistics ------------------------------------------
  double stats_start_time = MPI_Wtime();

  std::vector<openmp::Column_stats> stats =
      openmp::ComputeLocalStats(local_dataset, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  double stats_end_time = MPI_Wtime();

  // --- Phase 3: K-Means Clustering -----------------------------------------
  uint32_t num_clusters = 4;
  uint32_t max_iterations = 2000;

  double kmeans_start_time = MPI_Wtime();

  openmp::Centroids centroids =
      openmp::RunKMeans(local_dataset, num_clusters, max_iterations,
                        global_offset, total_rows, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  double kmeans_end_time = MPI_Wtime();
  double total_end_time = MPI_Wtime();

  // --- Results & Performance Report (rank 0 only) --------------------------
  if (rank == 0) {
    std::cout << "\n========================================\n";
    std::cout << "        HPC EXECUTION RESULTS\n";
    std::cout << "========================================\n";
    std::cout << "MPI Processes : " << num_procs << "\n";
    std::cout << "Total Rows    : " << total_rows << "\n";
    std::cout << "Total Columns : " << num_cols << "\n\n";

    std::cout << "--- 1. Column Statistics ---\n";
    for (uint32_t c = 0; c < num_cols; ++c) {
      std::cout << "Col " << c << " -> " << "Min: " << std::setw(8)
                << stats[c].min << " | " << "Max: " << std::setw(8)
                << stats[c].max << " | " << "Mean: " << std::setw(8)
                << stats[c].mean << " | " << "Var: " << std::setw(8)
                << stats[c].variance << "\n";
    }

    std::cout << "\n--- 2. K-Means Centroids ---\n";
    for (uint32_t k = 0; k < num_clusters; ++k) {
      std::cout << "Cluster " << k << " (Points: " << centroids.counts[k]
                << "): [ ";
      for (uint32_t c = 0; c < num_cols; ++c) {
        std::cout << centroids.GetValue(k, c)
                  << (c < num_cols - 1 ? ", " : " ");
      }
      std::cout << "]\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "             PERFORMANCE\n";
    std::cout << "========================================\n";
    std::cout << "I/O & Scatter : " << std::fixed << std::setprecision(4)
              << (io_end_time - io_start_time) << " seconds\n";
    std::cout << "Statistics    : " << (stats_end_time - stats_start_time)
              << " seconds\n";
    std::cout << "K-Means Core  : " << (kmeans_end_time - kmeans_start_time)
              << " seconds\n";
    std::cout << "Total Runtime : " << (total_end_time - total_start_time)
              << " seconds\n";
    std::cout << "========================================\n\n";
  }

  // --- Cleanup -------------------------------------------------------------
  MPI_Finalize();
  return EXIT_SUCCESS;
}