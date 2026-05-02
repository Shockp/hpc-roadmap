### Phase 1: The High-Performance Sequential Baseline (C++)
**Objective:** Establish a flawlessly correct and highly optimized foundation before introducing any parallel complexity. 

* **Project Initialization:** Set up the build environment (e.g., CMake) and enforce strict adherence to the Google C++ Style Guide to maintain a clean, professional codebase for the entire project lifecycle.
* **Memory Architecture:** Implement the grid using a single, linearized 1D array to guarantee contiguous memory allocation and maximize cache hits.
* **State Management:** Build the double-buffer system (T_old and T_new) to manage state updates without read-after-write hazards.
* **Boundary Logic:** Hardcode the fixed boundary conditions (100.0°C top, 0.0°C others) to act as constant "halo" values during stencil computation.
* **Logic Verification:** Run the simulation on a small 64x64 grid and print the output to visually confirm the heat is diffusing correctly over time.
* **Baseline Profiling:** Scale the grid to 4096x4096. Measure the execution time and cache miss rate; this is the definitive benchmark all future optimizations will be measured against.

---

### Phase 2: Multicore Scaling (OpenMP)
**Objective:** Exploit shared-memory parallelism on a single node to accelerate the core computation loops.

* **Loop Analysis:** Identify the critical nested loops executing the 5-point stencil over the grid.
* **Directive Implementation:** Apply `#pragma omp parallel for` to distribute loop iterations across available CPU threads.
* **Scheduling Experimentation:** Benchmark `static` versus `dynamic` scheduling strategies to see which handles the workload distribution best.
* **Overhead Mitigation:** Refactor the code to keep parallel regions open as long as possible, minimizing the overhead of spawning and destroying threads every iteration.
* **Cache Optimization:** Profile the thread execution to detect and eliminate any false sharing occurring at the memory boundaries between thread workloads.

---

### Phase 3: Multinode Scaling (MPI)
**Objective:** Distribute the memory and computational load across multiple distinct physical nodes using domain decomposition.

* **Domain Decomposition:** Partition the global matrix into smaller sub-grids (e.g., 1D row blocks or 2D Cartesian grids) assigned to individual MPI processes.
* **Halo Cell Setup:** Expand each local sub-grid by one row/column to store the boundary data belonging to adjacent processes.
* **Synchronous Communication:** Implement `MPI_Send` and `MPI_Recv` to exchange halo data between neighboring processes at the end of every time step.
* **Asynchronous Optimization:** Upgrade the communication layer to use non-blocking `MPI_Isend` and `MPI_Irecv`. 
* **Computation Overlap:** Restructure the execution order to compute the inner grid points while the non-blocking halo exchanges occur in the background, hiding network latency.

---

### Phase 4: Device Acceleration (CUDA)
**Objective:** Offload the intensely parallel stencil calculations to the GPU for massive throughput.

* **Data Transfer Architecture:** Allocate memory on the device (`cudaMalloc`) and implement the Host-to-Device data transfers for initialization.
* **Kernel Implementation:** Write the CUDA kernel to map threads to individual matrix elements to compute the stencil.
* **Memory Coalescing:** Ensure thread blocks access global memory in a coalesced manner to maximize memory bandwidth utilization.
* **Shared Memory Optimization:** Refactor the kernel to load tiles of the grid (along with their halos) into Shared Memory, drastically reducing repetitive global memory fetches.
* **Transfer Minimization:** Ensure the main simulation loop stays entirely on the GPU, transferring data back to the Host only when absolutely necessary (e.g., for writing final output).

---

### Phase 5: Hybridization & Exam Preparation
**Objective:** Combine methodologies and solidify your understanding of the architectural trade-offs to prepare for the final technical review.

* **Hybrid Integration (Optional but recommended):** Combine MPI with OpenMP or MPI with CUDA to utilize multiple GPUs across multiple nodes.
* **Final Benchmarking:** Run the fully optimized code on the 4096x4096 dataset and record the final execution times.
* **Metric Analysis:** Calculate the total speedup factor compared to your C++ baseline.
* **Code Defense Prep:** Review every optimization technique applied and prepare clear explanations for *why* a specific approach was chosen and *how* it impacted the hardware behavior.