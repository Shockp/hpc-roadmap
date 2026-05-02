### Phase 5: Hybridization & Final Evaluation Specification

This final document outlines the strategy for combining your parallelization layers and preparing for the project defense. To ensure you walk away with a highly professional foundation for future HPC work, all hybrid integrations must continue to strictly respect the Google C++ Style Guide, keeping complex distributed logic clean and maintainable.

---

### 1. Core Objectives
* **Architectural Synthesis:** Evolve the codebase into a true hybrid solution combining at least two technologies to attack different levels of parallelism (e.g., multicore + multinode, or multinode + GPU).
* **Maximum Throughput:** Achieve the absolute lowest execution time possible on the target hardware for the $4096 \times 4096$ dataset.
* **Defense Preparation:** The final evaluation relies entirely on an individual technical exam covering design, implementation, and metrics. To successfully pass, you must demonstrate a deep, personal understanding of every optimization applied.

### 2. Hybrid Architecture Strategies
You must select and implement at least one hybrid approach. The macro-architecture (node-to-node communication) will always be handled by MPI, while the micro-architecture (intra-node computation) is handled by either OpenMP or CUDA.

* **Option A: MPI + OpenMP (Distributed & Shared Memory)**
    * **Execution Model:** MPI spawns one process per physical node. Inside each node, OpenMP spawns multiple threads to utilize all local CPU cores.
    * **Thread Safety:** You must initialize MPI with threading support (e.g., `MPI_Init_thread` using `MPI_THREAD_FUNNELED` or `MPI_THREAD_MULTIPLE`). 
    * **Division of Labor:** The master thread of the OpenMP team handles the non-blocking MPI halo exchanges (`MPI_Isend`/`MPI_Irecv`), while the rest of the thread team computes the internal stencil points to maximize overlap.

* **Option B: MPI + CUDA (Multi-GPU Scaling)**
    * **Execution Model:** MPI manages the domain decomposition across multiple nodes, but the local sub-grid computation for each process is completely offloaded to a GPU.
    * **Device Assignment:** Ensure each MPI process correctly identifies and binds to a unique GPU using `cudaSetDevice()` to prevent multiple processes from bottlenecking on a single accelerator.
    * **Transfer Pipeline:** Implement a pipeline where the GPU computes the halo boundaries first, transfers *only* those boundary rows back to the Host (CPU), and then the Host exchanges them via MPI while the GPU continues computing the internal matrix.

### 3. Final Benchmarking & Metrics
You must generate a comprehensive performance report to defend your design choices during the exam.

* **The Baseline:** Use the Phase 1 sequential C++ execution time on the $4096 \times 4096$ grid as your $T_1$ (baseline time).
* **Speedup Calculation:** Calculate the strong scaling speedup ($S = \frac{T_1}{T_p}$) for every phase: OpenMP only, MPI only, CUDA only, and your final Hybrid solution.
* **Amdahl's Law Analysis:** Identify the remaining serial fractions in your code (e.g., memory allocation, file I/O, mandatory synchronization barriers) and document how they limit your maximum theoretical speedup.

### 4. Technical Exam Defense Checklist
Because no code or report is submitted, your grade depends entirely on your ability to verbally defend your architecture. Prepare clear, technical answers for the following critical areas:

* **Memory Hierarchy:** Be prepared to explain exactly why you chose a linearized 1D array over a 2D array, and how that decision impacted CPU L1/L2 cache hit rates.
* **OpenMP Scheduling:** Defend your choice of `static` vs. `dynamic` scheduling based on the uniform workload of the 5-point stencil.
* **Communication Overlap:** Be ready to draw or explain the exact sequence of events that allows your non-blocking MPI calls to hide network latency behind CPU/GPU computation.
* **CUDA Coalescing:** Explain how you mapped your 2D grid to CUDA blocks and how you ensured that memory accesses were coalesced within a warp.
* **Shared Memory Usage:** Quantify (conceptually) how much global memory traffic you saved by implementing shared memory tiling in your CUDA kernel.