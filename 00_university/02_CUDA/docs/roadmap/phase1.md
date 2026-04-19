### Phase 1: High-Performance Sequential Baseline Specification

This document details the exact technical requirements, memory architecture, and validation steps for the Phase 1 C++ baseline. To ensure a highly maintainable and professional foundation for the HPC pipeline, the codebase will strictly adhere to the Google C++ Style Guide, prioritizing extreme performance and cache efficiency.

---

### 1. Core Objectives
* **Establish Correctness:** Implement the mathematical model flawlessly to serve as the ground truth for all future parallel implementations.
* **Maximize Single-Thread Performance:** Design the memory layout to exploit CPU cache hierarchies (spatial and temporal locality).
* **Prepare for Parallelism:** Structure the simulation loop and state management so that introducing OpenMP, MPI, and CUDA requires minimal refactoring of the core logic.

### 2. Mathematical Model & Computation
The simulation relies on a simplified 5-point stencil. For every internal point in the grid, the new temperature at time $t+1$ is the arithmetic mean of its four direct neighbors at time $t$.

* **Update Formula:** $$T_{i,j}^{t+1}=\frac{T_{i-1,j}^{t}+T_{i+1,j}^{t}+T_{i,j-1}^{t}+T_{i,j+1}^{t}}{4}$$
* **Iteration Domain:** The formula is only applied to the *internal* grid points. Boundary points are treated as constants.

### 3. Memory Architecture
Dynamic memory allocation and multi-dimensional arrays (e.g., `double**`) cause memory fragmentation and cache misses. The baseline must enforce strict memory contiguity.

* **Matrix Linearization:** Allocate the $N \times N$ grid as a single, flat 1D array. 
    * *Access Pattern:* The 2D coordinate $(i, j)$ translates to the 1D index `[i * N + j]`.
* **Double Buffering:** Allocate two identical 1D arrays to prevent read-after-write dependencies.
    * `T_old`: The read-only state of the grid at the current time step $t$.
    * `T_new`: The write-only state of the grid for the next time step $t+1$.
    * *Swap:* At the end of each iteration, swap the pointers of `T_old` and `T_new` (do *not* copy the data) to prepare for the next step.

### 4. Boundary Conditions
The edges of the plate are constant heat sources/sinks and must never be overwritten during the stencil computation. 

| Boundary | Temperature Value | Behavior |
| :--- | :--- | :--- |
| **Top Edge** | 100.0 | Constant heat source. |
| **Bottom Edge** | 0.0 | Constant heat sink. |
| **Left Edge** | 0.0 | Constant heat sink. |
| **Right Edge** | 0.0 | Constant heat sink. |

*Note: When updating the internal points adjacent to the edges, the algorithm will naturally read these constant boundary values.*

### 5. Input Parameters
The executable must accept command-line arguments to dictate the simulation scope dynamically.

1.  **N (Grid Size):** The dimension of the $N \times N$ matrix.
2.  **Iterations:** The total number of time steps to simulate.

### 6. Validation & Benchmarking Protocol
Before moving to Phase 2, the baseline must pass the following two stages:

* **Stage 1: Logic Verification ($64 \times 64$)**
    * Run the simulation on a small grid.
    * Output the final state to a file or standard output to visually confirm that the heat from the top edge (100.0) is smoothly diffusing downward into the 0.0 zones.
* **Stage 2: Hardware Stress Test ($4096 \times 4096$)**
    * A $4096 \times 4096$ matrix of `double` floats requires $\approx 134$ MB of memory per buffer (well beyond standard L3 cache limits).
    * Measure the exact total execution time of the main simulation loop.
    * This timing is the definitive baseline. If a future OpenMP or MPI implementation is slower than this execution time, the parallelization overhead has outweighed the computational gains.