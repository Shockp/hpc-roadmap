# hpc-roadmap

A hands-on journey through High-Performance Computing—from modern C++ fundamentals to GPU-accelerated workloads. This repository tracks my progress through foundational texts and university-level HPC projects, focusing on performance optimization, distributed systems, and hardware-aware programming.

---

## 🚀 Featured Projects

While the roadmap follows a structured curriculum, the following projects represent the culmination of these concepts in complex, real-world scenarios:

### [2D Heat Diffusion Simulator](00_openmp_cuda_mpi/02_heat_diffusion)
A comprehensive study of HPC paradigms solving the 2D heat equation.
- **Solvers**: Sequential, OpenMP, MPI (Blocking & Non-blocking), CUDA, and Hybrid (CUDA + OpenMP).
- **Features**: Double buffering, advanced profiling (Setup/Compute/Comm), and automated benchmarking scripts.
- **Goal**: Compare scaling across multi-core, distributed, and GPU-accelerated environments.

### [Distributed K-Means Pipeline](00_openmp_cuda_mpi/01_distributed_kmeans)
A high-performance C++17 pipeline for clustering large-scale numerical datasets.
- **Architecture**: MPI-based data distribution (Scatter/Gather) with OpenMP-accelerated local computation.
- **Performance**: Optimized using array reductions and efficient data redistribution (`MPI_Alltoallv`).

---

## 🗺️ Roadmap & Curriculum

### Phase 1: Core Fundamentals & Hardware Awareness
Before writing extremely fast code, you need a solid grasp of modern C++ and exactly how the underlying hardware executes it.

- **A Tour of C++, 3rd Edition**
  - *Why here:* This is your primer. It gives you a fast, comprehensive overview of modern C++ features.

- **Computer Systems: A Programmer's Perspective**
  - *Why here:* Learn how code translates to assembly, how the CPU pipeline works, and crucially, how CPU caches operate.

### Phase 2: C++ Mastery & Single-Threaded Performance
Master the C++ language and learn how to extract maximum performance from a single CPU core.

- **Professional C++, 6th Edition**
  - *Why here:* Covers memory management, templates, and standard library features critical for writing robust systems.

- **Effective Modern C++, by Scott Meyers**
  - *Why here:* Focuses on "writing C++ the right way" (perfect forwarding, move semantics, smart pointers).

- **Optimized C++, by Kurt Guntheroth**
  - *Why here:* Learn to benchmark, profile, and optimize memory allocation and data structures.

### Phase 3: The OS Interface & Concurrency
Understand how the operating system handles resources and how to write code that safely executes on multiple threads.

- **Linux System Programming, 2nd Edition**
  - *Why here:* HPC clusters run on Linux. Understanding system calls, process management, and file I/O is essential.

- **C++ Concurrency in Action, Second Edition**
  - *Why here:* Covers the C++ memory model, atomics, mutexes, and lock-free programming.

### Phase 4: Scaling to HPC and Accelerators
Move from standard multi-threading to massive scale, utilizing specialized hardware and distributed systems.

- **Parallel and High Performance Computing**
  - *Why here:* Introduces industry standards like OpenMP, MPI, and vectorization.

- **Programming Massively Parallel Processors, 4th Edition**
  - *Why here:* Learn CUDA and adapt algorithms for massive SIMD execution on GPUs.

### Phase 5: Applied Domain Knowledge
- **Hands-On Machine Learning with C++**
  - *Why here:* Apply optimized, parallel C++ skills to real-world, highly demanding ML workloads.

---

## 📂 Project Structure

```text
.
├── 00_openmp_cuda_mpi/             # Practical HPC Implementations (University)
│   ├── 01_distributed_kmeans/      #   Distributed K-Means (MPI + OpenMP)
│   └── 02_heat_diffusion/          #   2D Heat Simulator (Seq, OMP, MPI, CUDA, Hybrid)
├── 01_fundamentals/                # Phase 1: Core Fundamentals
│   ├── 01_tour_of_cpp/             #   A Tour of C++
│   └── 02_cs_programmer_perspective/ #   CS:APP
├── 02_cpp_mastery/                 # Phase 2: C++ Mastery
│   ├── 01_professional_cpp/        #   Professional C++
│   ├── 02_effective_modern_cpp/    #   Effective Modern C++
│   └── 03_optimized_cpp/           #   Optimized C++
├── 03_os_and_concurrency/          # Phase 3: OS & Concurrency
│   ├── 01_linux_sys_programming/   #   Linux System Programming
│   └── 02_concurrency_in_action/   #   C++ Concurrency in Action
├── 04_parallel_and_gpu/            # Phase 4: Scaling & Accelerators
│   ├── 01_parallel_and_hpc/        #   Parallel & HPC
│   └── 02_prog_massiv_parallel_proc/ #   Programming Massively Parallel Processors
└── 05_applied_ml/                  # Phase 5: Applied Domain Knowledge
    └── 01_hands_on_ml/             #   Hands-On ML with C++
```

---

## 🛠️ Tech Stack & Tools
- **Languages:** C++17/20, CUDA C++
- **Build Systems:** CMake, Make
- **Parallelism:** OpenMP, MPI, `std::thread`, CUDA
- **Tooling:** Clangd, GDB, Valgrind
- **Performance:** Google Benchmark, Linux `perf`, NVIDIA Nsight Systems/Compute

