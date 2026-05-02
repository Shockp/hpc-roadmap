## High-Performance Heat Diffusion Simulation: System Architecture

This document outlines the software architecture for the heat diffusion simulation. The primary goal of this design is to create a robust, high-performance base that seamlessly scales from a sequential execution to a fully distributed, GPU-accelerated HPC pipeline.

---

### 1. Core Architectural Principles

To ensure extreme performance and long-term maintainability across all five phases of the project, the architecture is built on three foundational pillars:

* **Strict Memory Contiguity (Hardware-Aware Design):** The architecture absolutely forbids arrays of pointers (e.g., `double**`). All multi-dimensional grid data will be stored in flat, 1D arrays. This guarantees spatial locality, maximizes CPU cache line utilization, and allows for perfectly coalesced memory accesses when transferred to the GPU.
* **Google C++ Style Guide Compliance:** To build a professional base for HPC, the entire codebase will strictly adhere to the Google C++ Style Guide. By enforcing rigid naming conventions, precise scoping, and explicit memory management, the code remains highly readable and safe even as complex parallel layers (OpenMP threads, MPI communicators, CUDA kernels) are introduced.
* **Modular Separation of Concerns:** The core mathematical logic must be isolated from the parallelization mechanics. This prevents "spaghetti code" where MPI communication logic is entangled with the stencil computation, allowing you to swap execution models cleanly.

---

### 2. Proposed Directory Structure

A clean, standardized project layout is essential for an iterative HPC project managed with CMake.

```text
heat_diffusion/
├── CMakeLists.txt           # Build configuration and compiler flags (O3, OpenMP, MPI, CUDA)
├── include/
│   ├── config.h             # Structs for simulation parameters (N, iterations, boundary values)
│   ├── grid.h               # Grid memory management and inline coordinate mapping functions
│   ├── solver_seq.h         # Phase 1: Sequential solver logic
│   ├── solver_omp.h         # Phase 2: OpenMP solver logic
│   ├── solver_mpi.h         # Phase 3: MPI distributed solver logic
│   └── solver_cuda.cuh      # Phase 4: CUDA kernel definitions
├── src/
│   ├── main.cc              # Entry point, CLI parsing, and execution routing
│   ├── grid.cc              # Grid allocation, initialization, and double-buffer swapping
│   ├── solver_seq.cc        # Baseline implementation
│   ├── solver_omp.cc        # OpenMP implementation
│   ├── solver_mpi.cc        # MPI communication and domain logic
│   └── solver_cuda.cu       # GPU memory transfers and kernel execution
└── scripts/
    └── run_benchmarks.sh    # Script to execute all phases and capture timing metrics
```

---

### 3. Data Flow & State Management

The architecture dictates a strict **Double Buffer** pattern to manage the state of the grid over time, preventing read-after-write hazards during parallel execution.

| Component | Responsibility | Implementation Detail |
| :--- | :--- | :--- |
| `Grid State A` | Holds the temperatures at time $t$. | `std::vector<double>` or raw contiguous `double*`. |
| `Grid State B` | Computes and stores temperatures for time $t+1$. | Identically sized contiguous memory block. |
| `Buffer Swap` | Advances the simulation time step. | Swaps the underlying pointers. **Never** performs a deep copy of the array data. |

---

### 4. Parallel Abstraction Layers

To gracefully handle the incremental phases, the architecture uses specific strategies to encapsulate the parallel complexity:

* **Macro-Architecture (MPI):** The `solver_mpi` module acts as a wrapper around the core grid. It handles domain decomposition and allocates the required "halo" rows. The actual mathematical stencil remains identical to the sequential baseline; the MPI layer simply coordinates *when* the boundaries are updated and *which* sub-section of the grid is computed.
* **Micro-Architecture (OpenMP/CUDA):** The `solver_omp` and `solver_cuda` modules replace the inner nested loops of the sequential solver. The CUDA implementation specifically separates the Host logic (memory allocation, data transfer) from the Device logic (the `__global__` stencil kernel) to keep the host code clean.

---

### 5. Build and Compilation Strategy

The project will be orchestrated using CMake. This provides a unified way to handle the varying compilation requirements of the different phases:
* Enabling strict warnings (`-Wall -Wextra -Werror`) to catch undefined behavior early.
* Enforcing maximum compiler optimization (`-O3 -march=native`).
* Conditionally linking OpenMP libraries, MPI wrappers (`mpicxx`), and the CUDA compiler (`nvcc`) depending on the target being built.

Does this directory structure and architectural approach look solid to you, or would you like to dive directly into drafting the `CMakeLists.txt` and the `config.h` file to get the project started?