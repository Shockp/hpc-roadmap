#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include "grid.h"
#include "profiler.h"
#include "solver_seq.h"

#ifdef ENABLE_OPENMP
#include "solver_omp.h"
#endif
#ifdef ENABLE_MPI
#include <mpi.h>

#include "solver_mpi.h"
#endif
#ifdef ENABLE_CUDA
#include "solver_cuda.cuh"
#endif

int main(int argc, char** argv) {
  // 1. Command line Interface Parsing
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <grid_size_N> <iterations> <mode>\n"
              << "Modes: 'seq' (Sequential)"
#ifdef ENABLE_OPENMP
              << ", 'omp' (OpenMP)"
#endif
#ifdef ENABLE_CUDA
              << ", 'cuda' (CUDA)"
#endif
#ifdef ENABLE_MPI
              << ", 'mpi_blocking' (MPI Blocking), 'mpi_nonblocking' (MPI "
                 "Non-Blocking)"
#endif
              << "\nExample: " << argv[0] << " 4096 100 seq\n";
    return EXIT_FAILURE;
  }

  int n = 0;
  int iterations = 0;
  std::string mode = argv[3];

  try {
    n = std::stoi(argv[1]);
    iterations = std::stoi(argv[2]);
  } catch (const std::exception& e) {
    std::cerr << "Error: Invalid arguments. Please provide integers for N and "
                 "iterations.\n";
    return EXIT_FAILURE;
  }

  if (n <= 2 || iterations <= 0) {
    std::cerr << "Error: N must be > 2 and iterations must be > 0.\n";
    return EXIT_FAILURE;
  }

  // 2. Initialization
#ifdef ENABLE_MPI
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
  int rank = 0;
#endif

  if (rank == 0) {
    std::cout << "Initializing Heat Diffusion Simulation...\n"
              << "Grid Size: " << n << " x " << n << "\n"
              << "Iterations: " << iterations << "\n"
              << "Execution Mode: " << mode << "\n";
  }

  // 3. Execution & Profiling
  if (rank == 0) {
    std::cout << "Simulation started..." << std::endl;
  }

  heat_sim::ProfilerResult res;
  auto start_time = std::chrono::high_resolution_clock::now();

  // Route execution based on the chosen mode
  if (mode == "seq") {
    auto start_setup = std::chrono::high_resolution_clock::now();
    heat_sim::Grid grid(n);
    auto end_setup = std::chrono::high_resolution_clock::now();

    res = heat_sim::SolverSeq::Run(grid, iterations);
    res.setup_time +=
        std::chrono::duration<double>(end_setup - start_setup).count();
  }
#ifdef ENABLE_OPENMP
  else if (mode == "omp") {
    auto start_setup = std::chrono::high_resolution_clock::now();
    heat_sim::Grid grid(n);
    auto end_setup = std::chrono::high_resolution_clock::now();

    res = heat_sim::SolverOmp::Run(grid, iterations);
    res.setup_time +=
        std::chrono::duration<double>(end_setup - start_setup).count();
  }
#endif
#ifdef ENABLE_MPI
  else if (mode == "mpi_blocking") {
    res = heat_sim::SolverMpi::RunBlocking(n, iterations);
  } else if (mode == "mpi_nonblocking") {
    res = heat_sim::SolverMpi::RunNonBlocking(n, iterations);
  }
#endif
#ifdef ENABLE_CUDA
  else if (mode == "cuda") {
    auto start_setup = std::chrono::high_resolution_clock::now();
    heat_sim::Grid grid(n);
    auto end_setup = std::chrono::high_resolution_clock::now();

    res = heat_sim::SolverCuda::Run(grid, iterations);
    res.setup_time +=
        std::chrono::duration<double>(end_setup - start_setup).count();
  }
#endif
  else {
    std::cerr << "Error: Unknown execution mode '" << mode << "'.\n";
    return EXIT_FAILURE;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  res.total_time = std::chrono::duration<double>(end_time - start_time).count();

  // 4. Reporting
  if (rank == 0) {
    std::cout << "Simulation complete.\n"
              << "--- Profiling Results ---\n"
              << "Setup Time:       " << res.setup_time << " s\n"
              << "Compute Time:     " << res.compute_time << " s\n"
              << "Comm Time:        " << res.comm_time << " s\n"
              << "Total Time:       " << res.total_time << " s\n";
  }

#ifdef ENABLE_MPI
  MPI_Finalize();
#endif

  return EXIT_SUCCESS;
}