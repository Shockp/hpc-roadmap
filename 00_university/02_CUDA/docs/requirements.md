## Heat Diffusion Simulation: Project Requirements & Architectural Constraints

This document outlines the technical and architectural requirements for the High-Performance Computing (HPC) heat diffusion simulation[cite: 1]. The architecture will be built incrementally, establishing a highly optimized C++ baseline before integrating OpenMP, MPI, and CUDA[cite: 16, 31]. 

---

### 1. Evaluation & Delivery
* The project is assessed through an individual technical exam rather than source code or report submission[cite: 6, 7].
* The exam will cover specific design choices, implementation details, and obtained performance metrics[cite: 7].
* You must personally program and optimize the code to successfully answer the technical questions[cite: 8].

### 2. Core Mathematical Model
* The simulation models a square plate discretized into an $N \times N$ matrix[cite: 10].
* The computation utilizes a simplified 5-point stencil[cite: 13].
* The new temperature at a given point is the arithmetic mean of its four direct neighbors (top, bottom, left, right)[cite: 13].
* The specific formula to apply is $T_{i,j}^{t+1}=\frac{T_{i-1,j}^{t}+T_{i+1,j}^{t}+T_{i,j-1}^{t}+T_{i,j+1}^{t}}{4}$[cite: 14].

### 3. Baseline Implementation Constraints (C/C++)
To establish a robust foundation for HPC, the initial sequential implementation must adhere to strict performance criteria[cite: 16]. To maintain a clean and maintainable codebase throughout these optimizations, all C++ code must strictly adhere to the Google C++ Style Guide. 

* **Memory Layout:** The matrix must be linearized (allocated as a single contiguous 1D array) to maximize cache spatial locality and overall performance[cite: 18].
* **State Management:** Implement a Double Buffer pattern (using `T_old` and `T_new`) to prevent data hazards and read-after-write dependencies during the stencil update[cite: 19].
* **Boundary Conditions (Top):** The top edge must remain constant at 100.0°C[cite: 21, 24].
* **Boundary Conditions (Rest):** The bottom, left, and right edges must remain constant at 0.0°C[cite: 23, 24].
* **Boundary Interactions:** Points on the edges of the plate will read these constant values as their "exterior" neighbors[cite: 25].
* **Input Parameters:** The program must accept `N` (Matrix dimension) and `Iterations` (Number of time steps) via command-line arguments[cite: 27, 28, 29].

### 4. Parallelization & Optimization Strategies
The project demands an incremental approach, starting with isolated technologies before moving to hybrid parallelization combining multicore, multinode, and GPU acceleration[cite: 32].

* **OpenMP (Shared Memory):** Parallelize stencil loops using compiler directives after identifying critical compute regions[cite: 34].
* **OpenMP (Shared Memory):** You must analyze the performance impact of static versus dynamic thread scheduling[cite: 35].
* **OpenMP (Shared Memory):** The architecture must actively prevent false sharing and minimize the overhead of parallel region creation[cite: 36].
* **MPI (Distributed Memory):** Implement domain decomposition to divide the global matrix across multiple processes[cite: 37, 38].
* **MPI (Distributed Memory):** Establish a communication strategy for edge/halo synchronization at each iteration[cite: 38].
* **MPI (Distributed Memory):** Experiment with non-blocking communications (`MPI_Isend`/`MPI_Irecv`) to attempt to overlap computation with communication[cite: 39].
* **CUDA (Device Acceleration):** Offload the heavy stencil computation to the GPU device[cite: 40].
* **CUDA (Device Acceleration):** Optimize global memory accesses via memory coalescing[cite: 41].
* **CUDA (Device Acceleration):** Utilize Shared Memory to cache stencil neighbors and reduce global memory traffic[cite: 41].
* **CUDA (Device Acceleration):** Optimize the data transfers between the Host and the Device[cite: 42].

### 5. Execution & Testing Strategy
* **Validation Phase:** Begin with a small $64 \times 64$ grid to verify computational logic and boundary conditions[cite: 44].
* **Performance Phase:** Scale the simulation to a $4096 \times 4096$ grid[cite: 44].
* **Hardware Observation:** Using the $4096 \times 4096$ size ensures the dataset exceeds L3 cache capacity, allowing you to observe real-world hardware memory behavior[cite: 44].