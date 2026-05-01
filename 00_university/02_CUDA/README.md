# High-Performance Heat Diffusion Simulation

A high-performance 2D heat diffusion simulator implemented in C++17, CUDA, and MPI. This project serves as a comprehensive study of various High-Performance Computing (HPC) paradigms, ranging from single-threaded sequential execution to hybrid GPU-accelerated distributed systems.

The simulation solves the 2D heat equation using a finite difference method on an $N \times N$ grid, applying boundary conditions where the top edge is held at 100°C and other edges at 0°C.

### 🧮 Mathematical Model

The computation utilizes a 5-point stencil, where the temperature of a cell at the next time step ($t+1$) is the arithmetic mean of its four direct neighbors at the current time step ($t$):

$$T_{i,j}^{t+1}=\frac{T_{i-1,j}^{t}+T_{i+1,j}^{t}+T_{i,j-1}^{t}+T_{i,j+1}^{t}}{4}$$


## 🚀 Key Features

- **Multi-Paradigm Solvers**:
  - `seq`: Baseline sequential CPU implementation.
  - `omp`: Multi-core acceleration using **OpenMP**.
  - `mpi_blocking`: Distributed memory parallelization via **MPI** (Blocking communication).
  - `mpi_nonblocking`: Overlapping computation and communication via **MPI Non-Blocking** primitives.
  - `cuda`: Massive parallelization on NVIDIA GPUs using **CUDA kernels**.
  - `hybrid`: Advanced **CUDA + OpenMP** integration for multi-level parallelism.
- **Double Buffering**: O(1) buffer swapping to eliminate unnecessary memory copies between iterations.
- **Modern C++17**: Clean, type-safe architecture using modern standards.
- **Advanced Profiling**: Granular timing for Setup, Computation, and Communication phases.

## 🛠️ System Requirements

- **Compiler**: GCC 8+ or Clang 7+ (C++17 support required).
- **Build System**: CMake 3.18+.
- **CUDA** (Optional): NVIDIA GPU with Compute Capability 6.0+ and CUDA Toolkit 11.0+.
- **MPI** (Optional): OpenMPI or MPICH implementation.
- **OpenMP**: Supported by most modern C++ compilers.

## 🏗️ Compilation

Standard CMake build workflow:

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

To enable specific features, Ensure the dependencies (CUDA/MPI) are present in your environment. The `CMakeLists.txt` will automatically detect and enable available modules.

## 💻 Usage

Run the simulation by specifying the grid size, number of iterations, and the execution mode:

```bash
./heat_sim <grid_size_N> <iterations> <mode>
```

### Execution Modes

| Mode | Description |
| :--- | :--- |
| `seq` | Sequential CPU execution |
| `omp` | Multi-threaded CPU (OpenMP) |
| `mpi_blocking` | Distributed MPI (Blocking) |
| `mpi_nonblocking` | Distributed MPI (Non-blocking) |
| `cuda` | GPU Accelerated (CUDA) |
| `hybrid` | Multi-core CPU + GPU (CUDA + OpenMP) |

**Example**:
```bash
./heat_sim 4096 1000 cuda
```

> [!TIP]
> **Recommended Test Sizes**:
> - **Validation**: Use a $64 \times 64$ grid to quickly verify logic and boundary conditions.
> - **Performance**: Use a $4096 \times 4096$ grid to exceed L3 cache capacity and observe real hardware memory behavior.

For MPI modes, use `mpirun`:
```bash
mpirun -np 4 ./heat_sim 4096 1000 mpi_nonblocking
```

## 📁 Project Structure

- `src/`: Core implementation files (`.cc`, `.cu`).
- `include/`: Header files and simulation configurations.
- `scripts/`: Helper scripts for automated benchmarking.
- `docs/`: Technical requirements and architectural roadmaps.
- `CMakeLists.txt`: Modular build configuration.

## 📊 Benchmarking

A benchmarking script is provided in the `scripts/` directory to automate performance evaluation across different modes and grid sizes.

```bash
./scripts/run_benchmarks.sh
```