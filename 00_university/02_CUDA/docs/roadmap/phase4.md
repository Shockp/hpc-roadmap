### Phase 4: Device Acceleration (CUDA) Specification

This document outlines the architectural requirements and optimization strategies for the Phase 4 CUDA implementation. Transitioning from CPU-bound processing to GPU acceleration requires rethinking memory hierarchy and thread execution. To maintain a highly readable and professional HPC codebase, all Host (C++) and Device (CUDA) code must continue strictly adhering to the Google C++ Style Guide where applicable.

---

### 1. Core Objectives
* **Massive Parallelism:** The primary objective is to transfer the massive computational load of the grid to the GPU (Device).
* **Memory Bandwidth Optimization:** The kernel design must optimize access to global memory through coalesced accesses.
* **Latency Reduction:** The architecture must make use of Shared Memory to reduce data traffic when reading the stencil neighbors.
* **Transfer Efficiency:** Efficient management of data transfers between the Host and the Device will be strictly evaluated.

### 2. Memory Management & Transfer Strategy
The PCIe bus connecting the CPU (Host) to the GPU (Device) is a severe bottleneck. The architecture must minimize Host-Device communication.

* **Device Allocation:** Allocate the linearized 1D arrays (`T_old` and `T_new`) directly on the GPU global memory using `cudaMalloc`.
* **Initialization Transfer:** Use `cudaMemcpy` (Host-to-Device) to transfer the initial state of the grid only *once* before the simulation loop begins.
* **Persistent Device State:** The entire time-stepping loop (the `Iterations` loop) must execute entirely on the GPU. Swap the `T_old` and `T_new` device pointers directly within the Host-side simulation loop without transferring data back.
* **Final Extraction:** Use `cudaMemcpy` (Device-to-Host) only after all iterations are complete to retrieve the final computed matrix for validation or output.

### 3. Kernel Design & Global Memory Optimization
The core computational unit will be a CUDA Kernel executed by thousands of threads simultaneously.

* **Grid and Block Mapping:** Map the 2D computational grid to a 2D grid of CUDA thread blocks (e.g., using `dim3` for blocks of $16 \times 16$ threads). Each thread is responsible for calculating the new temperature of exactly one $i, j$ grid coordinate.
* **Coalesced Access:** Because C/C++ stores arrays in row-major order, ensure that the `x` dimension of the CUDA thread block corresponds to the innermost loop (the columns/j-index). This guarantees that adjacent threads in a warp access contiguous memory addresses, achieving full memory coalescing.
* **Boundary Masking:** Inside the kernel, implement conditional logic (`if (i > 0 && i < N-1 && ...)`) to ensure threads mapped to the constant boundary edges do not overwrite the fixed conditions (100.0°C and 0.0°C).

### 4. Shared Memory (Tiling) Architecture
Relying solely on Global Memory is inefficient because the 5-point stencil reads the same neighbor values multiple times across different threads. You must implement a "tiling" strategy.

* **Shared Memory Allocation:** Define a `__shared__` memory array within the kernel. The size of this array must be equal to the thread block dimensions *plus* an extra padding of 1 cell on all four sides to accommodate the halo neighbors.
* **Collaborative Loading:** Have each thread load its corresponding global grid value into the shared memory tile. Threads mapped to the edges of the block must also fetch the required halo values from global memory.
* **Synchronization:** Insert a `__syncthreads()` barrier to guarantee that the entire shared memory tile is fully populated before any computation begins.
* **Compute Phase:** Calculate the stencil arithmetic using *only* the fast shared memory values, drastically reducing the number of global memory reads. Write the final result directly to the `T_new` array in global memory.

### 5. Validation & Benchmarking Protocol
The GPU implementation must be tested to ensure correctness and to demonstrate the massive speedup capabilities of device acceleration.

* **Correctness Verification:** Execute the code on the small $64 \times 64$ grid. Retrieve the memory to the Host and assert that it perfectly matches the output of the Phase 1 sequential baseline.
* **Hardware Stress Test:** Run the simulation on the $4096 \times 4096$ dataset.
* **Performance Profiling:** * Measure the total execution time (including memory transfers).
    * Measure the kernel execution time *excluding* memory transfers.
    * Compare the speedup against the C++ baseline, OpenMP, and MPI implementations.