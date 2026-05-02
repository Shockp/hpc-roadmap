### Phase 3: Multinode Scaling (MPI) Specification

This document details the architectural design and execution strategy for the Phase 3 MPI (Message Passing Interface) implementation. Transitioning to distributed memory requires a fundamental shift in how the simulation data is managed. To maintain a robust HPC foundation, all code must continue to strictly adhere to the Google C++ Style Guide, ensuring that communication logic is encapsulated cleanly and does not degrade single-thread performance.

---

### 1. Core Objectives
* **Distributed Memory Parallelism:** Shift from a single shared memory space to a distributed model, focusing the design heavily on domain decomposition.
* **Data Distribution:** Divide the global $N \times N$ matrix among multiple independent processes, ensuring memory footprint scales with the number of nodes.
* **Latency Hiding:** Actively overlap network communication with CPU computation by leveraging advanced MPI features.

### 2. Domain Decomposition Strategy
Since C++ utilizes a row-major memory layout, a 1D row-wise block decomposition is the most cache-efficient approach for this specific stencil.

* **Sub-grid Allocation:** Divide the global $N \times N$ matrix horizontally. If `P` is the number of MPI processes, each process will allocate a local grid of size $(N/P) \times N$. 
* **Halo Cells (Ghost Rows):** To compute the stencil at the local boundaries, each process needs the row belonging to its immediate top and bottom neighbors. Allocate an extra top and bottom row for each local grid to store these "halo" values.
* **Boundary Rank Logic:** Process Rank 0 will apply the constant 100.0°C top boundary condition, while the last Rank (`P-1`) will apply the constant 0.0°C bottom condition. All processes apply the 0.0°C left and right conditions.

### 3. Communication Architecture
Synchronizing the edges of the plate at each iteration is the primary bottleneck in distributed HPC applications. You must implement two distinct communication strategies for comparison.

* **Baseline Sync (Blocking):** * Implement the initial synchronization using `MPI_Send` and `MPI_Recv` (or `MPI_Sendrecv`).
    * The process halts computation while the halo rows are exchanged over the network.
* **Optimized Async (Non-blocking):** * Upgrade the communication layer to use non-blocking functions like `MPI_Isend` and `MPI_Irecv`.
    * **Computation Overlap:** To successfully overlap communication and computation, restructure the local execution loop:
        1. Initiate non-blocking halo exchanges for the top and bottom rows.
        2. Compute the stencil for the *inner* rows of the local grid (which do not depend on the halos).
        3. Call `MPI_Waitall` to guarantee halo exchanges are complete.
        4. Compute the stencil for the *outer* boundary rows.

### 4. Memory Management & Integration
* **Local Linearization:** Just like the Phase 1 C++ baseline, the local sub-grid (including halo rows) must be allocated as a contiguous 1D array.
* **Avoid Global Assembly:** During the main time-stepping loop, *never* gather the entire matrix back to Rank 0. The simulation must run entirely distributed. Only gather the matrix at the very end of the simulation if file output or final verification is required.

### 5. Validation & Benchmarking Protocol
The MPI implementation must be benchmarked to demonstrate true multinode scaling.

* **Correctness Verification:** Run the simulation on a $64 \times 64$ grid across 2 and 4 processes. Gather the final matrix to Rank 0 and assert it matches the sequential Phase 1 baseline exactly.
* **Network Stress Test:** Run the simulation on the $4096 \times 4096$ grid.
* **Scaling Metrics:** * Measure execution time across 1, 2, 4, and 8 processes. 
    * Calculate the communication overhead versus computation time.
    * Document the specific speedup achieved by switching from blocking to non-blocking communications.