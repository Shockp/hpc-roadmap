## Heat Diffusion Simulation: Project Requirements & Architectural Constraints

This document outlines the technical and architectural requirements for the High-Performance Computing (HPC) heat diffusion simulation. The architecture will be built incrementally, establishing a highly optimized C++ baseline before integrating OpenMP, MPI, and CUDA. 

---

### 1. Evaluation & Delivery
* The project is assessed through an individual technical exam rather than source code or report submission.
* The exam will cover specific design choices, implementation details, and obtained performance metrics.
* You must personally program and optimize the code to successfully answer the technical questions.

### 2. Core Mathematical Model
* The simulation models a square plate discretized into an $N \times N$ matrix.
* The computation utilizes a simplified 5-point stencil.
* The new temperature at a given point is the arithmetic mean of its four direct neighbors (top, bottom, left, right).
* The specific formula to apply is $T_{i,j}^{t+1}=\frac{T_{i-1,j}^{t}+T_{i+1,j}^{t}+T_{i,j-1}^{t}+T_{i,j+1}^{t}}{4}$.

### 3. Baseline Implementation Constraints (C/C++)
To establish a robust foundation for HPC, the initial sequential implementation must adhere to strict performance criteria. To maintain a clean and maintainable codebase throughout these optimizations, all C++ code must strictly adhere to the Google C++ Style Guide. 

* **Memory Layout:** The matrix must be linearized (allocated as a single contiguous 1D array) to maximize cache spatial locality and overall performance.
* **State Management:** Implement a Double Buffer pattern (using `T_old` and `T_new`) to prevent data hazards and read-after-write dependencies during the stencil update.
* **Boundary Conditions (Top):** The top edge must remain constant at 100.0°C.
* **Boundary Conditions (Rest):** The bottom, left, and right edges must remain constant at 0.0°C.
* **Boundary Interactions:** Points on the edges of the plate will read these constant values as their "exterior" neighbors.
* **Input Parameters:** The program must accept `N` (Matrix dimension) and `Iterations` (Number of time steps) via command-line arguments.

### 4. Parallelization & Optimization Strategies
The project demands an incremental approach, starting with isolated technologies before moving to hybrid parallelization combining multicore, multinode, and GPU acceleration.

* **OpenMP (Shared Memory):** Parallelize stencil loops using compiler directives after identifying critical compute regions.
* **OpenMP (Shared Memory):** You must analyze the performance impact of static versus dynamic thread scheduling.
* **OpenMP (Shared Memory):** The architecture must actively prevent false sharing and minimize the overhead of parallel region creation.
* **MPI (Distributed Memory):** Implement domain decomposition to divide the global matrix across multiple processes.
* **MPI (Distributed Memory):** Establish a communication strategy for edge/halo synchronization at each iteration.
* **MPI (Distributed Memory):** Experiment with non-blocking communications (`MPI_Isend`/`MPI_Irecv`) to attempt to overlap computation with communication.
* **CUDA (Device Acceleration):** Offload the heavy stencil computation to the GPU device.
* **CUDA (Device Acceleration):** Optimize global memory accesses via memory coalescing.
* **CUDA (Device Acceleration):** Utilize Shared Memory to cache stencil neighbors and reduce global memory traffic.
* **CUDA (Device Acceleration):** Optimize the data transfers between the Host and the Device.

### 5. Execution & Testing Strategy
* **Validation Phase:** Begin with a small $64 \times 64$ grid to verify computational logic and boundary conditions.
* **Performance Phase:** Scale the simulation to a $4096 \times 4096$ grid.
* **Hardware Observation:** Using the $4096 \times 4096$ size ensures the dataset exceeds L3 cache capacity, allowing you to observe real-world hardware memory behavior.