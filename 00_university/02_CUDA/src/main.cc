#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include "grid.h"
#include "solver_seq.h"

#ifdef ENABLE_OPENMP
#include "solver_omp.h"
#endif
#ifdef ENABLE_MPI
#include <mpi.h>
#include "solver_mpi.h"
#endif

int main(int argc, char** argv) {
  // 1. Command line Interface Parsing
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <grid_size_N> <iterations> <mode>\n"
              << "Modes: 'seq' (Sequential)"
#ifdef ENABLE_OPENMP
              << ", 'omp' (OpenMP)"
#endif
#ifdef ENABLE_MPI
              << ", 'mpi_blocking' (MPI Blocking), 'mpi_nonblocking' (MPI Non-Blocking)"
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

  heat_sim::Grid grid(n);

  // 3. Execution & Profiling
  std::cout << "Simulation started..." << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();

  // Route execution based on the chosen mode
  if (mode == "seq") {
    heat_sim::SolverSeq::Run(grid, iterations);
  }
#ifdef ENABLE_OPENMP
  else if (mode == "omp") {
    heat_sim::SolverOmp::Run(grid, iterations);
  }
#endif
#ifdef ENABLE_MPI
  else if (mode == "mpi_blocking") {
    heat_sim::SolverMpi::RunBlocking(n, iterations);
  } else if (mode == "mpi_nonblocking") {
    heat_sim::SolverMpi::RunNonBlocking(n, iterations);
  }
#endif
  else {
    std::cerr << "Error: Unknown execution mode '" << mode << "'.\n";
    return EXIT_FAILURE;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_time - start_time;

  // 4. Reporting
  if (rank == 0) {
    std::cout << "Simulation complete.\n"
              << "Total Execution Time: " << elapsed_seconds.count()
              << " seconds.\n";
  }

#ifdef ENABLE_MPI
  MPI_Finalize();
#endif

  return EXIT_SUCCESS;
}